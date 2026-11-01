"""Dev only: methods d and e, no-SAM baselines, Spark crop verdicts on present floors and on random drawn boxes."""
import sys, random
sys.path.insert(0, "lab/tightseed")
from core import *
import dev1, dev5, seeds as sd, multi as mu, tread

S = dict(main=("main",))
for st in DEV:
    cl = dev1.loose_clusters(st); fr = frames(st); gr = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr]
    d = mu.method_d(f"{st}__d_960", st, cl, gr)
    rows, false = score(st, d); print(st, "d raw", summary(rows, S["main"]), [(r["stair"], r["floor"], r["present"], [c[3] for c in r["claims"]]) for r in rows if not r["ok"]], flush=True)
    rows, false = score(st, d, dev5.make_check(st, d, fr)); print(st, "d tread", summary(rows, S["main"]), flush=True)
    e = mu.method_e(f"{st}__e_960", st, cl, gr)
    rows, false = score(st, e); print(st, "e vote", summary(rows, S["main"]), [(r["stair"], r["floor"], r["present"]) for r in rows if not r["ok"]], flush=True)
    sb = sd.seed_boxes(st, cl, "snap", fr, grow=.15)
    rows, _ = score(st, mu.no_sam(st, sb, gr)); print(st, "nosam tread", summary(rows, S["main"]), [(r["stair"], r["floor"], r["present"]) for r in rows if not r["ok"]], flush=True)
    rows, _ = score(st, mu.no_sam_ink(st, sb, gr)); print(st, "nosam ink", summary(rows, S["main"]), [(r["stair"], r["floor"], r["present"]) for r in rows if not r["ok"]], flush=True)
# Spark verdicts: every main stair box on every floor of two dev stacks, plus 12 random drawn boxes from each
rng = random.Random(11)
jobs = []; lab = []
for st in ("bme_f", "habs_stelizabeths_center"):
    T = TRUTH[st]; fr = frames(st); gr = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr]
    SEED = json.load(open(ROOT / "lab/seed/truth.json")).get(st, [])
    for k, s in T["stairs"].items():
        for f in range(T["n"]):
            b = s["boxes"].get(str(f)) or next(iter(s["boxes"].values()))
            jobs.append((st, f"T{k}_{f}", fr[f], b)); lab.append((str(f) in s["boxes"], st, k, f))
    others = [x["box"] for x in SEED] + [b for s in T["stairs"].values() for b in s["boxes"].values()]
    n = 0
    while n < 12:
        f = rng.randrange(T["n"]); w, h = rng.choice([(b[2] - b[0], b[3] - b[1]) for s in T["stairs"].values() for b in s["boxes"].values()])
        x, y = rng.randint(100, 1800 - int(w)), rng.randint(100, 1300 - int(h)); b = [x, y, x + w, y + h]
        if any(q[0] < b[2] and b[0] < q[2] and q[1] < b[3] and b[1] < q[3] for q in others) or (gr[f][y:y + int(h), x:x + int(w)] < 180).mean() < .02:
            continue
        jobs.append((st, f"R{n}_{f}", fr[f], b)); lab.append((False, st, f"rnd{n}", f)); n += 1
with ThreadPoolExecutor(6) as ex:
    ans = list(ex.map(mu.spark_yes, jobs))
pos = [a for a, l in zip(ans, lab) if l[0]]; neg = [a for a, l in zip(ans, lab) if not l[0]]
print("spark check: says stair on present", sum(pos), "/", len(pos), "says stair on absent or random", sum(neg), "/", len(neg), [l for a, l in zip(ans, lab) if a and not l[0]], [l for a, l in zip(ans, lab) if not a and l[0]])
print("spent", round(m.spent("tightseed"), 4))
