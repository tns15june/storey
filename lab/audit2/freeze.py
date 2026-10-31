"""Freeze the audit's live tests before any audit2 model call. No API."""
import hashlib, json, time
from pathlib import Path
A = Path("/Users/tsharma/Github/Meta/data/lab/audit2")
L = [json.loads(x) for x in open("/Users/tsharma/Github/Meta/data/lab/ledger.jsonl") if x.strip()]
assert not [r for r in L if r["name"] == "audit2"], "audit2 already called a model"
rule = {
 "written": time.strftime("%Y-%m-%d %H:%M:%S %z"),
 "ledger_audit2_rows_at_write": 0,
 "truth": "Plant locations come from the frozen presence and tightseed truth files, chosen by the fixed rules below. No new labels. One labeller upstream.",
 "plant_targets_sha256": hashlib.sha256((A / "plant_targets.json").read_bytes()).hexdigest(),
 "L1_sam_repeat": "Same box prompts and same video recipe as the cached tracks, 3 repeats: P on bme_k and queens_royce1, Ppin on mml_abbot. Watch svc_SW f2, R_G f2, ext_stair f3. Any repeat at conf>=.5 on a truth-absent floor is a wrong pass.",
 "L2_sam_erasure": "Stacks bme_k, habs_va_bldg1, wfu_huffman, queens_royce1, leonard4. Floor 2. Every cls main stair present on floors 1, 2, 3. Hand boxes pinned on every present floor except floor 2, the best case for the recipe. Kinds clean, hard, soft. A mask at conf>=.5 hitting the box on the erased floor would cancel a true ending flag under tightseed step 6.",
 "L3_crop_repeat_hard": "presence crop prompt, cut, schema and effort, no cache, 3 repeats on every held-out planted cell whose frozen row was not blank. Arm C gives Found on a repeat if v>=2, ink>=.40, not blank, crop yes. Same places on the clean sheet, 2 repeats, as the control.",
 "L4_soft": "22 targets in plant_targets.json, plants.soft, ink and blank computed exactly as run_cells does at the clean row's registered place, crop 3 repeats. v>=2 is assumed here, so a Found is conditional on the runs still claiming the stair. Clean cell 2 repeats as control.",
 "L4b_compare": "3 compare runs, prompt and effort as frozen, one soft plant per stack, at most 4 stacks. Take plants with any Found repeat in L4 first, then fill in target order, at most one per building unless fewer than 4 buildings qualify. Arm C decided by lab/presence/rule.py.",
 "L5_agent": "Spark low on all 32 overlays once, Wonder low on the 8 route overlays once. Report verdict flips against the cached run.",
 "L6_concept": "SAM concept floor plan on n161, n52, n133, 3 repeats each.",
 "L7_exits": "EXIT_P, EXIT_S, 2 by 2 tiles at medium on sheets c161 and c162, 2 fresh reads, half-sheet tile rule. Right is exactly 1440 and 294.",
 "L8_scan": "Sheets bcoe2 and bcoe3. Render 300 dpi greyscale, rotate .4 degrees, blur sigma .8, gaussian noise sigma 8, JPEG 60. 12 tiles, frozen prompt, low effort, one run. Truth boxes mapped through the same rotation. Scored IoU>=.3, false no truth at IoU>=.1. Keep needs recall>=.8 and false<=1 per sheet.",
 "budget": "3 USD. L7 and L8 run only while audit2 spend is under 2.4 USD.",
 "no_navy": "No navy_b14 sheet goes to a model: the owner has not ruled on how it was fetched.",
}
p = A / "rule.json"
assert not p.exists(), "already frozen"
p.write_text(json.dumps(rule, indent=1))
print(hashlib.sha256(p.read_bytes()).hexdigest(), rule["written"])
