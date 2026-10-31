"""Held-out runs under the frozen rule: every seed method on every stack, all cached. Prints no scores."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import dev1, seeds as sd, multi as mu


def in_crop(st, b):
    x1, y1, x2, y2 = TRUTH[st]["crop"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return x1 <= cx <= x2 and y1 <= cy <= y2


def one(st):
    cl = dev1.loose_clusters(st); fr = frames(st); gr = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr]
    json.dump(cl, open(SPK / f"{st}_clusters.json", "w"))
    print(st, "clusters", len(cl), "spent", round(m.spent("tightseed"), 3), flush=True)
    track(f"{st}__hand_960", st, dev1.hand_seeds(st), res=960)
    track(f"{st}__hand_2000", st, dev1.hand_seeds(st), res=2000)
    allseeds = {}
    for name, method, kw in (("loose", "loose", {}), ("snap", "snap", dict(grow=.15)), ("sparkcrop", "sparkcrop", {})):
        sb = sd.seed_boxes(st, cl, method, fr, **kw)
        allseeds[name] = [[o, f, b, fb] for o, f, b, fb in sb]
        track(f"{st}__{name}_960", st, [(o, f, b) for o, f, b, _ in sb], res=960)
    sb = allseeds["snap"]
    track(f"{st}__snap_2000", st, [(o, f, b) for o, f, b, _ in sb], res=2000)
    vp = R / f"vec_{st}.json"
    if vp.exists():
        it = [(f, b) for f, b in json.load(open(vp)) if in_crop(st, b)]
        vc = cluster(it)
        vs = [(i + 1, c["seed_floor"], c["seed_box"]) for i, c in enumerate(vc)]
        allseeds["vector"] = [[o, f, b, False] for o, f, b in vs]
        track(f"{st}__vector_960", st, vs, res=960)
    json.dump(allseeds, open(SPK / f"{st}_seeds.json", "w"))
    mu.method_d(f"{st}__d_960", st, cl, gr)
    mu.method_e(f"{st}__e_960", st, cl, gr)
    print(st, "done spent", round(m.spent("tightseed"), 3), flush=True)


for st in HELD:
    one(st)
