"""Part 1: read each sheet with Spark (no filename shown): printed floor label, storey rank, main plan box.
Also one joint call per building with the sheets shuffled. Results are cached in data/lab/align/spark_sheets.json."""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, "lab")
import metalab as m

OUT = Path("data/lab/align")
index = json.load(open(OUT / "sheets/index.json"))
cache_path = OUT / "spark_sheets.json"
cache = json.load(open(cache_path)) if cache_path.exists() else {}

box = {"type": "object", "additionalProperties": False, "required": ["x1", "y1", "x2", "y2"],
       "properties": {k: {"type": "integer"} for k in ["x1", "y1", "x2", "y2"]}}
ONE = {"type": "object", "additionalProperties": False, "required": ["printed_title", "printed_floor_label", "storey_rank", "plan_box", "sheet_rotation"],
       "properties": {"printed_title": {"type": "string"}, "printed_floor_label": {"type": "string"}, "storey_rank": {"type": "integer"}, "plan_box": box,
                      "sheet_rotation": {"type": "integer", "enum": [0, 90, 180, 270]}}}
P_ONE = ("This is one sheet of an architectural drawing set. It may be in any language. Report: "
         "printed_title, the sheet or building title as printed. printed_floor_label, the floor name exactly as printed. "
         "storey_rank, an integer for that floor: basement or cellar -1, sub-basement -2, the lowest full storey above ground 0, the next one up 1, and so on. Attic or roof uses the next rank above the top storey. "
         "plan_box, the box around the walls of the main floor plan only: exclude the title block, sheet border, legend, notes, schedules, dimension strings, grid bubbles, key plans, detail insets and site work. "
         "sheet_rotation, how many degrees clockwise the sheet must turn so that the text reads upright. "
         "Coordinates are integers on a 0-1000 grid of this image, x from the left edge and y from the top edge.")
JOINT = {"type": "object", "additionalProperties": False, "required": ["bottom_to_top"],
         "properties": {"bottom_to_top": {"type": "array", "items": {"type": "integer"}}}}


def read_one(bid, i):
    key = f"{bid}/{i}"
    if key in cache:
        return key, cache[key]
    im = cv2.imread(str(OUT / f"sheets/{bid}/{i}.jpg"))
    r, info = m.spark(P_ONE, [im], schema=ONE, effort=EFFORT)
    return key, {"read": r, "info": info}


def joint(bid):
    key = f"joint/{bid}"
    if key in cache:
        return key, cache[key]
    n = len(index[bid])
    order = list(range(n))
    random.Random(bid).shuffle(order)
    ims = [cv2.imread(str(OUT / f"sheets/{bid}/{i}.jpg")) for i in order]
    p = (f"These are {n} sheets from one building's drawing set, numbered 1 to {n} in the order attached. Each shows one storey. "
         "List the sheet numbers from the lowest storey to the highest storey, using the floor names printed on the sheets.")
    r, info = m.spark(p, ims, schema=JOINT, effort=EFFORT)
    return key, {"shown": order, "read": r, "info": info}


EFFORT = sys.argv[2] if len(sys.argv) > 2 else None
jobs = sys.argv[1]
tasks = []
if jobs == "probe":
    tasks = [lambda: read_one("bme_f", 1)]
if jobs in ("all", "one"):
    tasks += [lambda b=b, i=r["i"]: read_one(b, i) for b, rows in index.items() for r in rows]
if jobs in ("all", "joint"):
    tasks += [lambda b=b: joint(b) for b in index]
with ThreadPoolExecutor(6) as ex:
    for key, v in ex.map(lambda t: t(), tasks):
        cache[key] = v
        print(key, json.dumps(v.get("read"), ensure_ascii=False)[:230], round(v["info"]["usd"], 4), round(v["info"]["seconds"], 1), flush=True)
json.dump(cache, open(cache_path, "w"), indent=1, ensure_ascii=False)
print("spent", round(m.spent("align"), 4))
