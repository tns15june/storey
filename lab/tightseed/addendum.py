"""Secondary arms declared after the freeze and before any held-out score was printed."""
import hashlib, json, time
from pathlib import Path
p = Path("data/lab/tightseed/rule_addendum.json")
A = {
 "declared_before_first_heldout_score": True,
 "arm_f": "per-stair local video: window = 3x the snap seed, min side 320 px, each frame scaled so the long side is 960, one object per video, same presence and hit rules. Tests whether SAM's own input resolution is the limit.",
 "comparator_spark_floors": "Round 1 one-call prompt (lab/bakeoff/e3_pdf_floors.py FLOOR_P) on the held-out frames, cropped to the plan window; a listed stair is matched to a truth stair by median centre distance under 120 px, as lab/audit/heldout_score.py did; reported beside SAM, not used by it.",
 "rule_unchanged": "rule.json stands; these arms do not alter it",
}
if p.exists():
    raise SystemExit("exists")
s = json.dumps(A, indent=1); p.write_text(s)
print(hashlib.sha256(s.encode()).hexdigest(), time.strftime("%Y-%m-%dT%H:%M:%S%z"))
