"""Follow-up on the tracker's weak pair: repeats, a longer prompt box, prompting the other floor, and a local crop around the stairs.

Run: LAB_NAME=modern LAB_BUDGET=4 LAB_TOTAL=60 .venv/bin/python lab/modern/track2.py
"""
import copy, json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "lab/modern")
import track as T

BASE, RECT = copy.deepcopy(T.GT), {k: list(v["rect"]) for k, v in T.P.items()}


def variant(name, id, frame="panel150", prompt=None, boxes=None, rect=None, prompt_level=None):
    def go():
        gt = copy.deepcopy(BASE[id])
        if prompt_level:
            gt["prompt"] = prompt_level
        for o, lv, b in boxes or []:
            gt["objs"][o][lv] = b
        return name, gt, rect
    return name, id, frame, go


JOBS = []
for i in range(3):
    JOBS.append(variant(f"repeat{i}", "mml_capitol"))
    JOBS.append(variant(f"repeat{i}", "mml_baxter"))
JOBS += [
    variant("long_prompt", "mml_capitol", boxes=[(3, "main", [1744, 640, 1807, 985])]),
    variant("prompt_upper", "mml_capitol", prompt_level="upper"),
    variant("long_prompt", "mml_baxter", boxes=[(1, "main", [1791, 603, 1854, 839]), (2, "main", [1864, 603, 1928, 839])]),
    variant("prompt_upper", "mml_baxter", prompt_level="upper"),
]
# one at a time because track.run reads the module-level GT and P
res = []
for name, id, frame, go in JOBS:
    _, gt, rect = go()
    T.GT[id] = gt
    r = T.run(id, frame, True)
    r["variant"] = name
    res.append(r)
    print(name, id, r["correct"], "/", r["pairs"], [(x["name"][:8], x["level"], "mask" if x["mask"] else "NONE", x.get("iou"), x.get("conf")) for x in r["rows"]], flush=True)
    T.GT[id] = copy.deepcopy(BASE[id])
(T.OUT / "track2.json").write_text(json.dumps(res, indent=1))
print("spent", T.m.spent("modern"))
