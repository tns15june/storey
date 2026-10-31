"""Fourth building for the plan-only exterior door prompt."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import exits, key as K

B = json.load(open(Path(__file__).parent / "labels_baxter.json"))
r = exits.eye("mml_baxter", K.ROOT / "data/plans/mml_baxter/THE-BAXTER.pdf", 2, B["box"], B["doors"], B["pt_per_m"], 150)
print(r)
(K.OUT / "exits_baxter.json").write_text(json.dumps(r, indent=1))
