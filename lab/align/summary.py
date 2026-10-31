"""Tables for the report, from the cached runs only. No API call."""
import collections
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg
import stack_sam as ss

runs = [json.load(open(p)) for p in sorted(glob.glob(str(reg.OUT / "sam/*.json")))]
scored = [dict(r, s=ss.score(r)) for r in runs]
print("runs", len(runs), "mean s", round(np.mean([r["seconds"] for r in runs]), 1))


def agg(rs):
    t = collections.Counter()
    for r in rs:
        for k in ("correct", "total", "hit", "miss", "wrong", "tn", "fp"):
            t[k] += r["s"][k]
    return t


print("\nby tier, all runs")
for tier in ("none", "rough", "well"):
    a = agg([r for r in scored if r["tier"] == tier])
    print(tier, dict(a), f"{100 * a['correct'] / a['total']:.1f}%")
print("\nbme_f factorial: correct of 15 by tier x repeat x res")
for tier in ("none", "rough", "well"):
    for rep in (1, 2, 4):
        print(tier, f"r{rep}", [next(r["s"]["correct"] for r in scored if r["bid"] == "bme_f" and r["tier"] == tier and r["rep"] == rep and r["res"] == res) for res in (960, 1440, 2000)],
              "seconds", [next(r["seconds"] for r in scored if r["bid"] == "bme_f" and r["tier"] == tier and r["rep"] == rep and r["res"] == res) for res in (960, 1440, 2000)])
print("\nby repeat, bme_f only, mean seconds and MB")
for rep in (1, 2, 4):
    rs = [r for r in scored if r["bid"] == "bme_f" and r["rep"] == rep]
    print(rep, dict(agg(rs)), round(np.mean([r["seconds"] for r in rs]), 1), round(np.mean([r["video_mb"] for r in rs]), 2))
print("\nby res, bme_f only")
for res in (960, 1440, 2000):
    rs = [r for r in scored if r["bid"] == "bme_f" and r["res"] == res]
    print(res, dict(agg(rs)), round(np.mean([r["seconds"] for r in rs]), 1), round(np.mean([r["video_mb"] for r in rs]), 2))
print("\nrecipe cells, per building, correct/total: none_r1_960 rough_r1_960 well_r1_960 well_r1_1440")
for b in ss.B:
    print(b, [next((f"{r['s']['correct']}/{r['s']['total']}" for r in scored if r["key"] == f"{b}_{k}"), None) for k in ("none_r1_960", "rough_r1_960", "well_r1_960", "well_r1_1440")])
for k in ("none_r1_960", "rough_r1_960", "well_r1_960", "well_r1_1440"):
    print(k, dict(agg([r for r in scored if r["key"].endswith("_" + k)])))

# confidence and distance, non-reference floors only
good, bad, dist, low = [], [], collections.defaultdict(list), []
for r in scored:
    ref, boxes, absent = ss.boxes_for(r["bid"])
    for key, mk in r["masks"].items():
        i, f = map(int, key.split("/"))
        if f == ref or mk["conf"] is None:
            continue
        b = boxes[i]
        d = float(np.linalg.norm(np.array(mk["ref_xy"]) - np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])))
        ((good if d <= ss.TOL and (i, f) not in absent else bad)).append(mk["conf"])
        if d <= ss.TOL:
            dist[r["tier"]].append(d)
        if mk["conf"] < 0.7:
            low.append((r["key"], key, mk["conf"], round(d)))
print("\nconf of correct masks n", len(good), "p1 p5 p50", np.percentile(good, [1, 5, 50]).round(3), "min", min(good))
print("conf of wrong masks", sorted(bad))
print("masks under conf 0.7:", len(low), collections.Counter((k[0].rsplit("_", 3)[0], k[1]) for k in low).most_common(8))
for t, d in dist.items():
    print("hit distance px", t, "median", round(np.median(d), 1), "p95", round(np.percentile(d, 95), 1), "max", round(max(d), 1))
print("\nspent", round(sum(json.loads(x)["usd"] for x in open(reg.OUT.parent / "ledger.jsonl") if json.loads(x)["name"] == "align"), 4))
