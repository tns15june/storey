"""Erase controls with the prompt on the floor where the stair is drawn longest. Run like track.py."""
import copy, json, sys
sys.path.insert(0, "lab/modern")
import track as T

BASE = copy.deepcopy(T.GT)
res = []
for id, erase in (("mml_capitol", ("main", 3)), ("mml_capitol", ("ground", 1)), ("mml_baxter", ("main", 2)), ("mml_baxter", ("main", 1))):
    gt = copy.deepcopy(BASE[id])
    gt["prompt"] = "upper"
    T.GT[id] = gt
    r = T.run(id, "panel150", True, erase)
    res.append(r)
    print(id, erase, r["correct"], "/", r["pairs"], [(x["name"][:8], x["level"], x["gt"][:4], "mask" if x["mask"] else "NONE", x.get("iou"), x.get("conf")) for x in r["rows"]], flush=True)
(T.OUT / "track3.json").write_text(json.dumps(res, indent=1))
print("spent", T.m.spent("modern"))
