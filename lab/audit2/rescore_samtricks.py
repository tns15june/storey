"""Independent rescore of samtricks items 1 and 4 from saved results and truth. No API, imports nothing from lab/samtricks. Run with -I."""
import json
from pathlib import Path

D = Path("data/lab/samtricks")
T1 = json.load(open(D / "truth/item1_viewports.json"))["sheets"]
T4 = json.load(open(D / "truth/item4_clouds.json"))
R1 = json.load(open(D / "item1.json"))
R4 = json.load(open(D / "item4.json"))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0


def match(preds, truth, thr):
    pairs = sorted(((iou(p, t), i, j) for i, p in enumerate(preds) for j, t in enumerate(truth)), reverse=True)
    up, ut, out = set(), set(), []
    for v, i, j in pairs:
        if v >= thr and i not in up and j not in ut:
            up.add(i)
            ut.add(j)
            out.append((i, j, v))
    return out


def nms(boxes):
    keep = []
    for b in sorted(boxes, key=lambda q: -(q[2] - q[0]) * (q[3] - q[1])):
        if all(iou(b, k) < .3 for k in keep):
            keep.append(b)
    return keep


def item4():
    print("ITEM 4 clouds, IoU>=.3 one-to-one, false = no truth at IoU>=.1")
    tiles_per_sheet = {}
    for run in ("spark_tiles", "spark_tiles2"):
        tot = dict(n=0, hit=0, false=0)
        for k, truth in T4["clouds"].items():
            preds = nms([c["box"] for c in R4[run][k]["clouds"]])
            tb = [c["box"] for c in truth]
            h = match(preds, tb, .3)
            hp = {i for i, _, _ in h}
            ht = {j for _, j, _ in h}
            fl = [p for i, p in enumerate(preds) if i not in hp and not any(iou(p, t) >= .1 for t in tb)]
            tot["n"] += len(tb)
            tot["hit"] += len(h)
            tot["false"] += len(fl)
            print(f"  {run:13s} {k}: {len(h)}/{len(tb)} false {len(fl)} preds {len(preds)} missed {[truth[j]['id'] for j in range(len(truth)) if j not in ht]} usd {R4[run][k].get('usd', 0):.3f}")
        print(" ", run, tot)
    print("  size of truth clouds in sheet points (w x h):", sorted(round((c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1])) for v in T4["clouds"].values() for c in v))
    both = [(k, c["id"]) for k, v in T4["clouds"].items() for c in v]
    print("  sheets", len(T4["clouds"]), "clouds", len(both), "all from one PDF (BCOE addendum 3)")


def item1():
    print("ITEM 1 viewports, conf floor .3, one-to-one IoU>=.5")
    keys = ["n161", "n79", "n133", "c11", "c16", "c7", "b2", "b9", "b14", "b18", "u2", "u9", "u11", "u7", "a8", "a9", "a11", "af2", "hv2", "bk2"]  # lab/samtricks/reg.py HELD
    print("  held-out keys", len(keys))
    def score(fn):
        hit = req = false = noplan_false = 0
        for k in keys:
            t = T1[k]
            preds = fn(k)
            h = match(preds, t["req"], .5)
            hp = {i for i, _, _ in h}
            fl = [p for i, p in enumerate(preds) if i not in hp and not any(iou(p, o) >= .5 for o in t["opt"])]
            hit += len(h)
            req += len(t["req"])
            false += len(fl)
            noplan_false += bool(fl) and not t["req"]
        return dict(hit=hit, req=req, false=false, noplan_sheets_with_false=noplan_false, noplan_sheets=sum(not T1[k]["req"] for k in keys))
    print("  SAM  ", score(lambda k: [x["box"] for x in R1[k]["sam"] if (x["conf"] or 0) >= .3]))
    print("  Spark", score(lambda k: [R1[k]["spark"]]))
    print("  Left 78", score(lambda k: [[0, 0, 780, 1000]]))
    print("  Ink  ", score(lambda k: [R1[k]["ink"]]))
    print("  zero-box sheets by truth:", {k: len(T1[k]["req"]) for k in keys if not [x for x in R1[k]["sam"] if (x["conf"] or 0) >= .3]})
    print("  all SAM confs held-out:", sorted(round(x["conf"], 3) for k in keys for x in R1[k]["sam"])[:12], "...")


item4()
item1()
