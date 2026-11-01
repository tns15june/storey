"""Freeze what the end-to-end run and the letter test are scored against, before any model call on these sheets. No API."""
import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, "lab/agent")
import letter_data as L
D = Path("data/lab/agent")
doc = {
    "_written": time.strftime("%Y-%m-%d %H:%M:%S %z"), "building": "nibs_clinic", "labeller": "one labeller by eye, not a licensed reviewer",
    "changes_to_saved_data": [
        "Seed A: treads of the south stair removed from sheets 059 and 162, matrix says absent on floor 2",
        "Seed B: matrix wrongly says the central stair is absent on floor 2, the sheet still shows it",
        "Seed C: dimension row printed 2875 given measured 2663, the only 'differs' row that is planted",
    ],
    "must_file": {"A": "South stair missing on floor 2 (kind stair_continuity, sheet 059 or 162)", "C": "dimension row D10, printed 2875 against measured 2663"},
    "must_not_file": {"B": "Central stair missing on floor 2: a false absence, sheet 059 shows it",
                      "service": "Service stair on floor 1 only: an exterior stair to the service yard, correct as drawn",
                      "mispairs": {"D04": "1500/988", "D08": "1030/256", "D31": "2420/8357", "D33": "1763/1147", "D37": "2910/2080"}},
    "not_scored_but_watched": ["route from 1E27 BENCHSTOCK leaves through a window gap; its 15.33 m is a real Storey mistake",
                               "marked travel distances 41.75 and 46.65 m against Storey routes: a finding here is judged by eye, real but not seeded",
                               "any finding not listed above is judged by eye and reported"],
    "invented": "a finding is invented if it cites a call id that does not exist, or a number found in no tool text and not legible on a sheet it viewed; numbers read from images are checked by eye",
    "letter": {"findings": L.FINDINGS, "header": L.HEADER, "traps": ["finding 3 has no printed value", "finding 5 has no section", "the style excerpt carries sections and numbers that must not leak", "the task gives no permit number, date or address"],
               "rule": "every number, sheet id and section in the letter must occur in the findings; list numbering does not count", "baseline": "letter built in code from the findings, adds nothing by construction"},
}
p = D / "e2e_truth.json"
p.write_text(json.dumps(doc, indent=1))
print("sha256", hashlib.sha256(p.read_bytes()).hexdigest(), doc["_written"])
