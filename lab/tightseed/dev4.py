"""Dev only: snap grow setting, Spark crop seeds, then SAM on both. Thresholds chosen here get frozen in rule.json."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import dev1, seeds as sd, tread


def seed_iou(st, cl, sb):
    r = []
    for c, (o, f, b, fb) in zip(cl, sb):
        own = [k for k in assign(st, c["seed_box"]) if TRUTH[st]["stairs"][k]["cls"] != "amb"]
        for k in own:
            t = TRUTH[st]["stairs"][k]["boxes"].get(str(f))
            if t:
                r.append((st, k, f, round(iou(b, t), 2), fb))
    return r


if __name__ == "__main__":
    allr = {}
    for st in DEV:
        cl = dev1.loose_clusters(st); fr = frames(st)
        for name, kw in (("loose", {}), ("snap15", dict(grow=.15)), ("snap30", dict(grow=.3)), ("snap60", dict(grow=.6)), ("snap100", dict(grow=1.0)), ("sparkcrop", {})):
            sb = sd.seed_boxes(st, cl, "snap" if name.startswith("snap") and name != "sparkcrop" else name, fr, **kw)
            allr.setdefault(name, []).extend(seed_iou(st, cl, sb))
    for k, v in allr.items():
        a = np.array([x[3] for x in v])
        print(k, "n", len(a), "median", round(float(np.median(a)), 2), ">=.5", int((a >= .5).sum()), "fallback", sum(x[4] for x in v), v if k in ("snap60", "sparkcrop") else "")
    print("spent", round(m.spent("tightseed"), 4))
