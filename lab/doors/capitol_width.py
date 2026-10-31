"""Clear width check on a second vector building: arc radius at 1/4 in = 1 ft against the printed door schedule."""
import json, sys
from pathlib import Path
import fitz, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import vec_arcs as v

ROOT = Path(__file__).resolve().parents[2]
PT_PER_IN = 18 / 12  # 1/4 in = 1 ft means 18 pt per foot
out = {}
for name, f, p, box, sched in (("capitol", "mml_capitol/THE-CAPITOL.pdf", 3, (1600, 300, 2100, 1100), "marks 1,2,4,8 are 36x80 swing doors"),
                               ("baxter", "mml_baxter/THE-BAXTER.pdf", 2, (1555, 155, 2180, 1075), "marks 1,2,4,8,9,10,12,13?,16: swing doors 36x80 by schedule")):
    pg = fitz.open(str(ROOT / "data/plans" / f))[p]
    s = v.page_segments(pg)
    # whole page: other views on the sheet are at larger scales, so their arcs fall outside this radius band
    a = v.attach_leaf(v.find_arcs(s, 24 * PT_PER_IN, 40 * PT_PER_IN, smin=70), s)
    seen = []
    for x in a:
        if not any(np.hypot(x["cx"] - y["cx"], x["cy"] - y["cy"]) < 3 for y in seen):
            seen.append(x)
    r = [{"in": round(x["r"] / PT_PER_IN, 1), "sweep": round(x["sweep"]), "leaf": bool(x["leaf"]), "xy": [round(x["cx"]), round(x["cy"])]} for x in seen]
    out[name] = {"schedule": sched, "arcs": r}
    print(name, len(r), [q["in"] for q in r])
(ROOT / "data/lab/doors/capitol_width.json").write_text(json.dumps(out, indent=1))
