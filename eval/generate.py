"""Generates a test drawing set and its truth.

    python -m eval.generate --building 710 --seed 1 --variant clean

A variant is clean, mixed (about ten defects), second (the mixed set resubmitted) or the name of one defect kind. Output goes to data/generated/<building>_<seed>_<variant>/: set.pdf and truth.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

from shapely.geometry import Polygon

from storey.rules import JURISDICTIONS, rule

from . import defects, model
from .draw import draw_set, plan_frame
from .swiss import ROOT

OUT = ROOT / "data/generated"
# One stair that stays in place, 2 to 4 units on every floor, residential throughout, walls on a square grid, a street door in the hall. 2828 and 6267 are the ones whose halls check 14 can compare with a printed width.
BUILDINGS = {"710": 5, "3370": 5, "5615": 5, "16281": 5, "3128": 5, "2828": 4, "6267": 6}   # building and stories
MIXED = "stair_mid", "stair_discharge", "index_extra", "floor_name", "stories", "area_total", "tag_no_row", "door_type", "exit_capacity", "room_name"
CORRECTED = "stair_mid", "floor_name", "stories", "area_total", "room_name"   # each correction lands on the spot its comment points at, which is where the second review looks
PX_PER_PT = {"stage files": 6000 / 2592, "review.json": 2000 / 2592}   # a finding's box in each: stage files keep ingest's render of a 36 by 24 in sheet, and review.json the copy the screen draws on
PARTITION_M = 0.12
DRAWN = "stair_mid", "stair_discharge", "corridor"   # kinds that point at the drawing and not at a row of text


def check_11(m):
    """What check 11 should say on each floor above discharge, by jurisdiction, from the rule rows. The set declares R-2 with sprinklers, and its one stair is on each floor unless a defect took it off."""
    required = rule("ibc.1006.3.3", "houston", occupant_load=m.load_total["load"])["value"]
    out = []
    for city in JURISDICTIONS:
        floors = []
        for story, f in enumerate(m.floors[1:], 2):
            one_stair = rule("ibc.1006.3.4.t1", city, sprinklers="NFPA 13", story=story)["value"] != "NP"
            floors.append({"floor": f.name, "story": story, "outcome": "one stair depends on conditions" if one_stair and f.stair_on_plan else "too few stairs"})
        out.append({"jurisdiction": city, "exits_required": required, "floors": floors})
    return out


def _digest(truth):
    return hashlib.sha256(json.dumps({k: v for k, v in truth.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest()


def verify(path):
    """True if a truth file still matches the hash it carries."""
    truth = json.loads(Path(path).read_text())
    return truth.get("sha256") == _digest(truth)


def _boxed(rows, anchors):
    return [{k: v for k, v in row.items() if k != "anchor"} | {"box": [round(float(v), 1) for v in anchors[row["sheet"]][row["anchor"]]]} for row in rows]


def _truth(m, pdf, rows, anchors, **extra):
    truth = {"building": m.building, "seed": m.seed, "variant": m.variant, "project": m.name, "source": "Swiss Dwellings v3.0.0, CC BY 4.0", "box_units": "sheet points", "px_per_pt": PX_PER_PT,
             "sheets": [{"number": s.number, "title": s.title, "kind": s.kind, "floor": s.floor} for s in m.sheets if s.in_set], "defects": _boxed(rows, anchors), "check_11": check_11(m),
             "pdf_sha256": hashlib.sha256(Path(pdf).read_bytes()).hexdigest(), **extra}
    return truth | {"sha256": _digest(truth)}


def _partition(m):
    """The change no cloud mentions: a partition added across a bedroom on the top floor's plan. Returns where it is and its wording."""
    f = m.floors[-1]
    room = max((r for r in f.rooms if r.name == "BEDROOM"), key=lambda r: r.poly.area)
    x0, y0, x1, y1 = room.poly.bounds
    y = y0 + (y1 - y0) * 0.8
    wall = Polygon([(x0, y), (x1, y), (x1, y + PARTITION_M), (x0, y + PARTITION_M)])
    f.plan_walls.append(wall)
    return f"A1.{f.number}", [round(v, 1) for v in plan_frame(m).rect(wall)], f"A partition now crosses room {room.number}, and no cloud marks it."


def _cloud(comment):
    """A cloud round a correction: loose on a drawing, and tight round text, because a scallop stands 0.4 of its pitch past the box and must keep off the next row and the title strip's edges. Pad and pitch are in points."""
    pad, scallop = (14, 26) if comment["kind"] in DRAWN else (3, 14)
    x0, y0, x1, y1 = comment["box"]
    return {"sheet": comment["sheet"], "box": [x0 - pad, y0 - pad, x1 + pad, y1 + pad], "kind": comment["kind"], "scallop": scallop}


def second(building, seed, pdf):
    """The mixed set resubmitted: some defects corrected with clouds round them, one change no cloud mentions, and every other sheet as it was."""
    first = model.load(building, seed, "mixed")
    planted = [defects.plant(first, kind) for kind in MIXED]
    comments = _boxed(planted, draw_set(first, pdf))
    m = model.load(building, seed, "second")
    still = [defects.plant(m, kind) for kind in MIXED if kind not in CORRECTED]
    sheet, box, wording = _partition(m)
    m.revision = {"number": 1, "date": "11/15/2026", "text": "RESPONSE TO COMMENTS"}
    fixed = draw_set(m, pdf)   # a cloud goes round the correction as it now stands, which can be wider than the error was
    m.clouds = [_cloud(c) for c in _boxed([p for p in planted if p["kind"] in CORRECTED], fixed)]
    anchors = draw_set(m, pdf)
    rows = [{"kind": c["kind"], "check": c["check"], "sheet": c["sheet"], "box": c["box"], "state": "Looks resolved" if c["kind"] in CORRECTED else "Still open"} for c in comments]
    rows.append({"kind": "partition", "check": 15, "sheet": sheet, "box": box, "state": "New change", "wording": wording})
    return _truth(m, pdf, still, anchors, parent=f"{building}_{seed}_mixed", comments=rows, clouds=m.clouds, changed_sheets=sorted({c["sheet"] for c in m.clouds} | {sheet}),
                  revision_rows=[{"sheet": s, "box": anchors[s]["revision"]} for s in sorted({c["sheet"] for c in m.clouds})])


def make(building, seed, variant):
    """Writes one set and returns its folder."""
    folder = OUT / f"{building}_{seed}_{variant}"
    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder / "set.pdf"
    if variant == "second":
        truth = second(building, seed, pdf)
    else:
        m = model.load(building, seed, variant)
        kinds = {"clean": (), "mixed": MIXED}.get(variant, (variant,))
        rows = [defects.plant(m, kind) for kind in kinds]
        truth = _truth(m, pdf, rows, draw_set(m, pdf))
    (folder / "truth.json").write_text(json.dumps(truth, indent=1))
    return folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variant", default="clean", choices=["clean", "mixed", "second", *defects.KINDS])
    a = ap.parse_args()
    print(make(a.building, a.seed, a.variant))


if __name__ == "__main__":
    main()
