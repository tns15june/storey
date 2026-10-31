"""Freeze the post-hoc recipe before it touches the fresh stack. Refuses to overwrite."""
import hashlib, json, time
from pathlib import Path
p = Path("/Users/tsharma/Github/Meta/data/lab/tightseed/rule_posthoc.json")
A = {
 "status": "POST HOC: chosen after held-out scores were seen. rule.json is unchanged and its scores are reported first.",
 "recipe_P": "Spark DOC boxes per floor, Round 1 cluster, medoid floor; then two rounds of zoomed Spark crops with post1.WHOLE_P: crop 3x box min side 300, then 1.8x the answer min side 200, long side 1000, zoom at most 5x; nearest box to the current centre; keep the last answer, or the current box if Spark returns nothing; SAM on the seed floor at 960, repeat 1, conf>=.5",
 "pin_variant": "same box on every floor the cluster has a member",
 "why_changed": "held-out failures were centre error of 15 to 45 px on stairs 30 to 100 px wide, and flight-only seeds losing the track; hand boxes were whole-stair boxes, not tread-only boxes",
 "fresh_stack": "royce_east, truth_fresh.json, never sent to a model before this file",
 "fresh_arms": ["hand", "loose", "P", "Ppin", "Spark verdict at the P box on every floor, no SAM"],
}
if p.exists():
    raise SystemExit("exists")
s = json.dumps(A, indent=1); p.write_text(s)
print(hashlib.sha256(s.encode()).hexdigest(), time.strftime("%Y-%m-%dT%H:%M:%S%z"))
