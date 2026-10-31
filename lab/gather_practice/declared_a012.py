"""Recompute the tables an applicant declared on a real exit plan, from its vector text alone.

Sheet A012, The Center for Learning and Resilience (Bid Addendum 3, page 2).
Run: /Users/tsharma/Github/Meta/.venv/bin/python -I /Users/tsharma/Github/Meta/lab/gather_practice/declared_a012.py
Writes data/lab/gather_practice/declared_a012.json. No model is used.
"""
import collections
import json
import math
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent.parent
PDF = ROOT / "data/reference/practice/bcoe_center_addendum3.pdf"
ROOM = re.compile(r"^(\d+[A-Z]?) ([A-Z].*?) (\d+) SF(?: (\d+) (\d+))?$")
EXIT = re.compile(r"^EXIT (\d) (\d+) (\d+) (\d+)(?: \d)? ([\d.]+) OK$")
CORR = re.compile(r"^(\d+[A-Z]?) ([A-Z /]+) (\d+) ([\d.]+) ([\d ]+(?:/\d+)?)\" OK$")


def rows(words, x0, x1, y0, y1):
    out = collections.defaultdict(list)
    for w in sorted((w for w in words if x0 < w[0] < x1 and y0 < w[1] < y1), key=lambda a: a[0]):
        out[round(w[1] / 6)].append(w[4])
    return [" ".join(out[k]) for k in sorted(out)]


def merge_split_rows(lines):
    # shortcut: two rows on this sheet sit a few points apart so their words land in neighbouring buckets
    fixed, i = [], 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i] == "50 SF 300 1" and lines[i + 1] == "107 STORAGE":
            fixed.append("107 STORAGE 50 SF 300 1")
            i += 2
        elif lines[i] == "1 114 IT 80 SF 150":
            fixed.append("114 IT 80 SF 150 1")
            i += 1
        else:
            fixed.append(lines[i])
            i += 1
    return fixed


def main():
    with fitz.open(PDF) as doc:
        page = doc[1]
        words = page.get_text("words")
        paths = len(page.get_drawings())
    lines = merge_split_rows(rows(words, 1990, 2420, 300, 970))
    rooms = [m.groups() for m in map(ROOM.match, lines) if m]
    grand = next(int(x) for x in re.findall(r"Grand total (\d+) SF", " ".join(lines)))
    declared_total = int(re.search(r"Grand total \d+ SF (\d+)", " ".join(lines)).group(1))
    assert len(rooms) == 50, len(rooms)

    mismatch = []
    area_sum = occ_sum = non_load_area = 0
    for num, name, area, factor, occ in rooms:
        area_sum += int(area)
        if factor is None:
            non_load_area += int(area)
            continue
        occ_sum += int(occ)
        expect = math.ceil(int(area) / int(factor))
        if expect != int(occ):
            mismatch.append({"room": num, "name": name, "area": int(area), "factor": int(factor), "declared": int(occ), "recomputed": expect})

    exit_lines = rows(words, 1990, 2420, 1840, 1925)
    exits = [m.groups() for m in map(EXIT.match, exit_lines) if m]
    assert len(exits) == 6, exit_lines
    exit_rows = [{"exit": int(e), "clear_in": int(c), "occupants": int(o), "required_in": float(r), "recomputed_in": round(int(o) * 0.2, 1)} for e, _, c, o, r in exits]
    corr_lines = rows(words, 1990, 2420, 1340, 1436)
    corridors = [m.groups() for m in map(CORR.match, corr_lines) if m]
    assert len(corridors) == 7, corr_lines

    example = re.search(r"(\d+) OCUPANTS \* 0\.2\" = ([\d.]+)\" REQUIRED \(X\) 1 DOOR", " ".join(rows(words, 1990, 2420, 1950, 1990)))
    result = {
        "vector_paths_on_sheet": paths,
        "rooms_parsed": len(rooms),
        "area_sum_sf": area_sum,
        "area_declared_grand_total_sf": grand,
        "area_rooms_without_a_load_sf": non_load_area,
        "area_with_load_sf": area_sum - non_load_area,
        "occupants_sum": occ_sum,
        "occupants_declared_total": declared_total,
        "rooms_where_ceiling_differs": mismatch,
        "exits": exit_rows,
        "exit_occupant_sum": sum(e["occupants"] for e in exit_rows),
        "exit_table_exit_1_occupants": exit_rows[0]["occupants"],
        "worked_example_exit_1_occupants": int(example.group(1)),
        "worked_example_exit_1_required_in": float(example.group(2)),
        "corridor_rows_parsed": len(corridors),
    }
    out = ROOT / "data/lab/gather_practice/declared_a012.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
