"""Rows of one Swiss Dwellings building. geometries.csv is 1 GB, so it is streamed from the zip once and the rows kept are cached."""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data/plans/swiss_dwellings/swiss-dwellings-v3.0.0.zip"
SOURCE = ROOT / "data/generated/source"
FIELDS = ("floor_id", "plan_id", "unit_id", "unit_usage", "entity_type", "entity_subtype", "geometry", "elevation", "height")


def fetch(buildings):
    """Streams the csv once and caches each building's rows. A building's rows are not contiguous in the file, so the whole file is read."""
    csv.field_size_limit(sys.maxsize)
    found = {b: {} for b in buildings}
    with zipfile.ZipFile(ZIP) as z, z.open("swiss-dwellings-v3.0.0/geometries.csv") as raw:
        for row in csv.DictReader(io.TextIOWrapper(raw, "utf8")):
            if row["building_id"] in found:
                found[row["building_id"]].setdefault((row["floor_id"], row["entity_subtype"], row["geometry"]), {k: row[k] for k in FIELDS})   # a shared wall or door is listed once per unit
    SOURCE.mkdir(parents=True, exist_ok=True)
    for b, rows in found.items():
        if not rows:
            raise KeyError(f"No building {b} in Swiss Dwellings")
        (SOURCE / f"{b}.json").write_text(json.dumps(list(rows.values())))


def rows(building):
    path = SOURCE / f"{building}.json"
    if not path.exists():
        fetch([building])
    return json.loads(path.read_text())
