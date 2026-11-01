"""Dev only: Spark grounding, hand-box tracks at 960 and 2000, loose tracks, and a probe of one object id on two frames."""
import sys
sys.path.insert(0, "lab/tightseed")
from core import *


def hand_seeds(stack, cls=("main", "short", "ext")):
    out = []
    for i, (k, s) in enumerate(TRUTH[stack]["stairs"].items()):
        if s["cls"] not in cls:
            continue
        fl = sorted(int(f) for f in s["boxes"])
        f = fl[(len(fl) - 1) // 2]
        out.append((i + 1, f, s["boxes"][str(f)]))
    return out


def loose_clusters(stack):
    n = TRUTH[stack]["n"]
    with ThreadPoolExecutor(5) as ex:
        sp = list(ex.map(ground_floor, [(stack, f) for f in range(n)]))
    items = [(f, b) for f in range(n) for b in sp[f]["boxes"]]
    return cluster(items)


if __name__ == "__main__":
    for st in DEV:
        for res in (960, 2000):
            tr = track(f"{st}__hand_{res}", st, hand_seeds(st), res=res)
            rows, false = score(st, tr)
            print(st, "hand", res, summary(rows, ("main",)))
            print("   bad", [(r["stair"], r["floor"], r["present"], [c[3] for c in r["claims"]]) for r in rows if not r["ok"]])
        cl = loose_clusters(st)
        seeds = [(i + 1, c["seed_floor"], c["seed_box"]) for i, c in enumerate(cl)]
        print(st, "clusters", len(cl), [(c["seed_floor"], [round(v) for v in c["seed_box"]], c["floors"]) for c in cl])
        for res in (960, 2000):
            tr = track(f"{st}__loose_{res}", st, seeds, res=res)
            rows, false = score(st, tr)
            print(st, "loose", res, summary(rows, ("main",)), "false tracks", len(false))
            print("   bad", [(r["stair"], r["floor"], r["present"], [c[3] for c in r["claims"]]) for r in rows if not r["ok"]])
    # probe: same object id prompted on two frames
    st = "bme_f"
    s = TRUTH[st]["stairs"]["I"]["boxes"]
    try:
        tr = track(f"{st}__probe_multi", st, [(1, 0, s["0"]), (1, 2, s["2"])], res=960)
        print("multi-frame accepted", {k: {f: (v["box"], v["conf"]) for f, v in o.items()} for k, o in tr["tracks"].items()})
    except Exception as e:
        print("multi-frame rejected", repr(e)[:400])
    print("spent", round(m.spent("tightseed"), 4))
