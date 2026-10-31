"""Count buildings, floors and stair records in Swiss Dwellings v3 without unpacking the zip.

Run: /Users/tsharma/Github/Meta/.venv/bin/python -I lab/gather_data/swiss_stats.py
Output: data/lab/gather_data/swiss_stats.json
"""
import collections
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
csv.field_size_limit(sys.maxsize)
z = zipfile.ZipFile(ROOT / "data/plans/swiss_dwellings/swiss-dwellings-v3.0.0.zip")
rows = csv.DictReader(io.TextIOWrapper(z.open("swiss-dwellings-v3.0.0/geometries.csv"), encoding="utf-8"))
print(rows.fieldnames)
types = collections.Counter()
floors = collections.defaultdict(set)
stairs = collections.defaultdict(lambda: collections.defaultdict(int))
n = 0
for r in rows:
    n += 1
    types[(r["entity_type"], r["entity_subtype"])] += 1
    key = (r["site_id"], r["building_id"])
    floors[key].add(r["floor_id"])
    if "STAIR" in r["entity_subtype"].upper():
        stairs[key][r["floor_id"]] += 1
hist = collections.Counter(len(v) for v in floors.values())
out = {"rows": n, "buildings": len(floors), "floors_per_building": dict(sorted(hist.items())),
       "buildings_3plus_floors": sum(len(v) >= 3 for v in floors.values()),
       "buildings_3plus_floors_with_stairs_on_every_floor": sum(len(v) >= 3 and len(stairs[k]) == len(v) for k, v in floors.items()),
       "entity_counts": {f"{a}/{b}": c for (a, b), c in types.most_common(60)}}
(ROOT / "data/lab/gather_data/swiss_stats.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "entity_counts"}))
print(sorted(out["entity_counts"].items(), key=lambda kv: -kv[1])[:30])
