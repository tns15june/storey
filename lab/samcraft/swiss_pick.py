"""Pull a few multi-floor buildings with columns, corridors and stairs out of Swiss Dwellings, without unpacking it."""
import csv
import io
import json
import zipfile

from data import OUT, ROOT

csv.field_size_limit(10**9)
z = zipfile.ZipFile(ROOT / "data/plans/swiss_dwellings/swiss-dwellings-v3.0.0.zip")
r = csv.DictReader(io.TextIOWrapper(z.open("swiss-dwellings-v3.0.0/geometries.csv"), "utf8"))
keep = {"WALL", "COLUMN", "DOOR", "ENTRANCE_DOOR", "STAIRS", "CORRIDOR", "STAIRCASE", "SHAFT", "ELEVATOR", "WINDOW", "RAILING", "LOBBY", "CORRIDORS_AND_HALLS"}
cur, rows, out = None, [], {}


def good(rows):
    fl = {}
    for x in rows:
        fl.setdefault(x["floor_id"], set()).add(x["entity_subtype"])
    ok = [f for f, s in fl.items() if {"COLUMN", "CORRIDOR", "STAIRCASE", "WALL"} <= s or {"COLUMN", "STAIRCASE", "WALL", "SHAFT"} <= s]
    return len(ok) >= 4 and sum(x["entity_subtype"] == "COLUMN" for x in rows) >= 24


for i, x in enumerate(r):
    if x["building_id"] != cur:
        if rows and good(rows):
            out[cur] = rows
            print("building", cur, len(rows), flush=True)
            if len(out) == 4:
                break
        cur, rows = x["building_id"], []
    if x["entity_subtype"] in keep:
        rows.append({k: x[k] for k in ("plan_id", "floor_id", "entity_type", "entity_subtype", "geometry", "elevation")})
print(i, "rows read", len(out))
(OUT / "swiss_subset.json").write_text(json.dumps(out))
