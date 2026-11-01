"""Spark proposes stairs, shafts, exit doors and corridors on every aligned sheet. Output seeds_raw.json plus overlays."""
import sys, json
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'lab'); sys.path.insert(0, 'lab/defects')
import metalab as m
from common import *
box = {"type": "object", "additionalProperties": False, "required": ["label", "x_min", "y_min", "x_max", "y_max"], "properties": {"label": {"type": "string"}, "x_min": {"type": "integer"}, "y_min": {"type": "integer"}, "x_max": {"type": "integer"}, "y_max": {"type": "integer"}}}
arr = {"type": "array", "items": box}
schema = {"type": "object", "additionalProperties": False, "required": ["sheet_floor_label", "stairs", "shafts", "exit_doors", "corridors"], "properties": {"sheet_floor_label": {"type": "string"}, "stairs": arr, "shafts": arr, "exit_doors": arr, "corridors": arr}}
P = ("You are an object grounding expert reading one architectural floor plan. Provide bounding boxes for every instance of: "
     "\"stairs\" (each flight or stair enclosure with treads, interior or exterior), \"shafts\" (elevators, lifts, dumbwaiters, duct or pipe shafts, chimney flues drawn as small enclosures), "
     "\"exit_doors\" (doors in the exterior wall leading outside), \"corridors\" (main long corridors). Do not miss any. "
     "Coordinates are integers on a 0-1000 grid of this image, x from the left edge, y from the top edge. Use the floor label as printed on the sheet for sheet_floor_label, or empty if none is printed.")
def one(b, f):
    p = OUT / 'stack' / b / f"{f['level']}.png"
    r, info = m.spark(P, [p], schema=schema)
    return b, str(f['level']), r, info
jobs = [(b, f) for b in BUILDINGS for f in CAT[b]['floors']]
res = {}
with ThreadPoolExecutor(6) as ex:
    for b, l, r, info in ex.map(lambda a: one(*a), jobs):
        res.setdefault(b, {})[l] = r
        print(b, l, r['sheet_floor_label'], len(r['stairs']), len(r['shafts']), len(r['exit_doors']), len(r['corridors']), round(info['usd'], 4), round(info['seconds']), flush=True)
json.dump(res, open(OUT / 'seeds_raw.json', 'w'), indent=0)
C = {'stairs': (0, 0, 255), 'shafts': (255, 0, 200), 'exit_doors': (0, 170, 0), 'corridors': (255, 150, 0)}
for b in res:
    for l, r in res[b].items():
        im = cv2.imread(str(OUT / 'stack' / b / f'{l}.png')); h, w = im.shape[:2]
        for k, c in C.items():
            for i, o in enumerate(r[k]):
                p1 = (o['x_min'] * w // 1000, o['y_min'] * h // 1000); p2 = (o['x_max'] * w // 1000, o['y_max'] * h // 1000)
                cv2.rectangle(im, p1, p2, c, 2); cv2.putText(im, f"{k[:2]}{i}", (p1[0], p1[1] - 3), 0, .6, c, 2)
        cv2.imwrite(str(OUT / 'thumbs' / f'prop_{b}_{l}.jpg'), im)
