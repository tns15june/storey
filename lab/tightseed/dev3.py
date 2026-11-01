"""Dev only, no API: tread score on stairs, on empty floors, and on random non-stair boxes, to place the check threshold."""
import sys, json, random
sys.path.insert(0, "lab/tightseed")
from core import *
import tread, vis
SEED = json.load(open(ROOT / "lab/seed/truth.json"))
rng = random.Random(7)
pos, neg, rnd = [], [], []
for st in DEV:
    T = TRUTH[st]; fr = frames(st); gr = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr]
    for k, s in T["stairs"].items():
        for f in range(T["n"]):
            if f in s.get("exclude", ()):
                continue
            b = s["boxes"].get(str(f)) or next(iter(s["boxes"].values()))
            r = tread.score_box(gr[f], b)
            (pos if str(f) in s["boxes"] else neg).append((st, k, f, round(r["cover"], 3), round(r["q90"], 2), round(r["period"], 1), r["steps"]))
    # random boxes away from every labelled stair of any class; seed truth boxes are in seed frames for capitol only, close enough to exclude generously
    all_boxes = [b for s in T["stairs"].values() for b in s["boxes"].values()]
    all_boxes += [x["box"] for x in SEED.get(st, [])]
    cx1, cy1, cx2, cy2 = T["crop"]
    sizes = [(b[2] - b[0], b[3] - b[1]) for s in T["stairs"].values() for b in s["boxes"].values()]
    for f in range(T["n"]):
        n = 0
        while n < 12:
            w, h = rng.choice(sizes)
            x, y = rng.randint(cx1, cx2 - int(w) - 1), rng.randint(cy1, cy2 - int(h) - 1)
            b = [x, y, x + w, y + h]
            if any(iou(b, q) > 0 or (q[0] < b[2] and b[0] < q[2] and q[1] < b[3] and b[1] < q[3]) for q in all_boxes):
                continue
            if (gr[f][y:y + int(h), x:x + int(w)] < 180).mean() < .02:  # blank paper is the easy case, keep boxes with drawing in them
                continue
            r = tread.score_box(gr[f], b)
            rnd.append((st, f, [round(v) for v in b], round(r["cover"], 3), round(r["q90"], 2), round(r["period"], 1), r["steps"]))
            n += 1
print("pos", len(pos)); print(sorted(pos, key=lambda r: r[3])[:14])
print("neg stair absent floors", neg)
print("rnd", len(rnd)); print(sorted(rnd, key=lambda r: -r[3])[:14])
c = np.array([r[3] for r in pos]); q = np.array([r[4] for r in pos]); cr = np.array([r[3] for r in rnd]); qr = np.array([r[4] for r in rnd])
for t in (.02, .04, .06, .08, .1, .12, .15, .2):
    print("cover>=", t, "pos kept", round((c >= t).mean(), 3), "rnd kept", round((cr >= t).mean(), 3))
for t in (.2, .3, .4, .5, .6, .7):
    print("q90>=", t, "pos kept", round((q >= t).mean(), 3), "rnd kept", round((qr >= t).mean(), 3))

sp = np.array([r[6] for r in pos]); sr = np.array([r[6] for r in rnd])
print("pos steps sorted", sorted(sp.tolist())); print("rnd steps top", sorted(sr.tolist())[-24:])
for t in (3, 4, 5, 6, 7, 8):
    for qt in (0, .3, .4):
        print("steps>=", t, "q90>=", qt, "pos", round(((sp >= t) & (q >= qt)).mean(), 3), "rnd", round(((sr >= t) & (qr >= qt)).mean(), 3), [r for r in rnd if r[6] >= t and r[4] >= qt][:3] if t == 6 and qt == .4 else "")
