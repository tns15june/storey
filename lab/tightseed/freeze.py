"""Freeze the decision rule from dev before any held-out model call. Refuses to overwrite."""
import hashlib, json, time
from pathlib import Path
p = Path("/Users/tsharma/Github/Meta/data/lab/tightseed/rule.json")
RULE = {
 "frozen_before_any_heldout_call": True,
 "dev_stacks": ["bme_f", "habs_stelizabeths_center", "mml_capitol"],
 "heldout_stacks": ["bme_k", "habs_va_bldg1", "wfu_huffman", "queens_royce1", "leonard4", "mml_abbot"],
 "presence": "mask area>0 and conf>=0.5, Round 1 rule",
 "hit": "lab/seed/common.py hit() with grow .25, as lab/audit/heldout_score.py used",
 "strict": "a stair is seeded when at least one track's seed hits its box on any floor; a present floor is right only if every such track has a mask that hits; an absent floor is right only if no such track claims it; no best-track choice",
 "scored": "cls main and short; ext reported separately; amb never scored",
 "video": "repeat 1, 960 px long side, crop to the stack's plan window, propagation both. Secondary, not for the recommendation unless stated: 2000 px",
 "loose_seed": "Round 1 DOC prompt per floor, Round 1 cluster, medoid floor",
 "hand_seed": "truth box on the lower-median present floor of the stair",
 "snap_seed": {"grow": 0.15, "thr": 0.34, "pmin": 3.0, "min_std": 4.0, "fallback": "loose box when snap finds nothing"},
 "sparkcrop_seed": {"window": "2.2x box, min side 220, up to 4x zoom, long side 1000", "prompt": "seeds.CROP_P", "fallback": "loose box when Spark returns nothing", "several_boxes": "union"},
 "vector_seed": "vec.stair_runs per floor, Round 1 cluster, medoid; wfu_huffman and mml_abbot only",
 "method_d": "one object id pinned on every floor where the cluster has a member, box = that member's snap, fallback member box",
 "method_e": "one object per cluster member; a floor is claimed when at least half of the cluster's tracks have a mask there; box = median",
 "tread_check": {"steps_min": 5, "q90_min": 0.4, "on": "mask box, pad 8 px"},
 "spark_check": {"prompt": "multi.CHECK_P", "window": "1.6x box, min side 200, red rectangle drawn on the crop"},
 "no_sam": ["seed box, unmoved, tread check on every floor", "seed box, ink ratio >= 0.45 of the seed floor", "Spark per-floor cluster membership"],
 "seed_iou": "IoU of a track's seed box with the hand box of the stair on the seed floor; unseeded stairs count as 0",
 "chosen_on_dev": "snap grow .15 over .6 (habs central snapped to the wrong run at .6, ghost returned on bme_f); res 960 equal to 2000 on hand seeds; steps>=5 and q90>=.4 kept 40/41 true stairs and 5/144 random drawn boxes, two of those five being real unlabelled stairs",
}
if p.exists():
    raise SystemExit("rule.json exists")
s = json.dumps(RULE, indent=1); p.write_text(s)
h = hashlib.sha256(s.encode()).hexdigest(); t = time.strftime("%Y-%m-%dT%H:%M:%S%z")
(p.parent / "rule.sha256").write_text(f"{h}  rule.json  written {t}\n")
print(h, t)
