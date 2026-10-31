"""Secondary arm f: one local video per cluster, so SAM sees the stair large. Declared in rule_addendum.json before any held-out score."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *


def window(b, shape, k=3.0, minside=320):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    w, h = max(k * (b[2] - b[0]), minside), max(k * (b[3] - b[1]), minside)
    return [max(0, int(cx - w / 2)), max(0, int(cy - h / 2)), min(shape[1], int(cx + w / 2)), min(shape[0], int(cy + h / 2))]


def one(a):
    st, o, f, b, fr = a
    tr = track(f"{st}__armf_{o}", st, [(1, f, b)], res=960, frs=fr, crop=window(b, fr[0].shape), upscale=True)
    return str(o), f, b, tr["tracks"].get("1", {})


def run(st):
    p = SAMD / f"{st}__armf.json"
    if p.exists():
        return json.load(open(p))
    fr = frames(st)
    sb = json.load(open(SPK / f"{st}_seeds.json"))["snap"]
    # only clusters seeded on a truth stair: spend control, false tracks are counted in the whole-sheet arms
    jobs = [(st, o, f, b, fr) for o, f, b, _ in sb if [k for k in assign(st, b) if TRUTH[st]["stairs"][k]["cls"] != "amb"]]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, jobs))
    o = dict(seeds=[[int(i), f, b] for i, f, b, _ in res], tracks={i: t for i, _, _, t in res}, res=960)
    json.dump(o, open(p, "w"))
    return o


if __name__ == "__main__":
    for st in HELD:
        o = run(st)
        rows, false = score(st, o)
        print(st, "armf", summary(rows), flush=True)
    print("spent", round(m.spent("tightseed"), 4))
