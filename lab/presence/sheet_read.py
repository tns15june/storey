"""Step 1. Printed floor label and plan box per sheet, plus the joint floor order, with the prompts of lab/align/sheet_read.py at the same low effort.
That script runs on import and writes to the align folder, so the prompts are copied here. Reads already paid for in the align cache are reused and their cost still counts."""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2

import sys; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/presence")
import pres
from pres import OUT, SH, INDEX, T
import metalab as m

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
P_JOINT = ("These are {n} sheets from one building's drawing set, numbered 1 to {n} in the order attached. Each shows one storey. "
           "List the sheet numbers from the lowest storey to the highest storey, using the floor names printed on the sheets.")
EFFORT = "low"
p = OUT / "spark_sheets.json"
cache = json.load(open(p)) if p.exists() else {}
old = json.load(open(pres.ROOT / "data/lab/align/spark_sheets.json"))
for k, v in old.items():
    if k.split("/")[-1 if k.startswith("joint/") else 0] in T:
        cache.setdefault(k, v)


def read_one(b, i):
    im = cv2.imread(str(SH / f"{b}/{i}.jpg"))
    r, info = m.spark(P_ONE, [im], schema=ONE, effort=EFFORT)
    return f"{b}/{i}", {"read": r, "info": info}


def joint(b):
    n = len(INDEX[b])
    order = list(range(n))
    random.Random(b).shuffle(order)
    if order == sorted(order):
        order = order[::-1]
    ims = [cv2.imread(str(SH / f"{b}/{i}.jpg")) for i in order]
    r, info = m.spark(P_JOINT.format(n=n), ims, schema=JOINT, effort=EFFORT)
    return f"joint/{b}", {"shown": order, "read": r, "info": info}


tasks = [lambda b=b, i=r["i"]: read_one(b, i) for b in T for r in INDEX[b] if f"{b}/{r['i']}" not in cache]
tasks += [lambda b=b: joint(b) for b in T if f"joint/{b}" not in cache]
print("tasks", len(tasks), flush=True)
with ThreadPoolExecutor(5) as ex:
    for k, v in ex.map(lambda t: t(), tasks):
        cache[k] = v
        print(k, json.dumps(v["read"], ensure_ascii=False)[:200], round(v["info"]["usd"], 4), flush=True)
json.dump(cache, open(p, "w"), indent=1, ensure_ascii=False)
for b in T:
    j = cache[f"joint/{b}"]
    got = [j["shown"][q - 1] for q in j["read"]["bottom_to_top"]]
    print(f"{b:26s} order {got} catalog_ok {got == list(range(len(INDEX[b])))} labels", [cache[f"{b}/{r['i']}"]["read"]["printed_floor_label"] for r in INDEX[b]])
print("spent", round(m.spent("presence"), 4))
