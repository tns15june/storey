"""Planted defects. Each edits the model before drawing, so a defect is an ordinary drafting error on the sheet, never a patch over a picture.

Each kind returns the sheet and the anchor truth points at, the wording of what is wrong, and a phrase of the title the finding should carry. `CHECKS` gives each kind's check and the severities the spec's table allows for it.
"""
import math
import random

import numpy as np
from shapely.geometry import Polygon

from .model import IN_PER_M, Sheet

OFFICE = "OFFICE"       # shares no word start with BEDROOM, because room_same forgives near names
NARROWED_IN = 16        # past the 12 in that printed widths may run over clear width
MAX_PRINTED_IN = 80     # the widest printed corridor width check 14 reads
STUB_M = 0.7            # half the length of the wall that moved


class NotPlantable(ValueError):
    """This building's geometry cannot hold this defect."""


def _floor(m, share):
    return m.floors[min(int(len(m.floors) * share), len(m.floors) - 1)]


def _plan(f):
    return f"A1.{f.number}"


def _interior(rng, f):
    return rng.choice(sorted((d for d in f.doors if d.kind == "interior"), key=lambda d: d.mark))


def stair_mid(m, rng):
    f = _floor(m, 0.5)
    f.stair_on_plan = False
    return _plan(f), "stair", f"The stair is left off {f.name}, and the walls round it stay.", "Stair"


def stair_discharge(m, rng):
    f = m.floors[0]
    f.stair_on_plan = False
    return _plan(f), "stair", f"The stair is left off {f.name}, the level of exit discharge.", "Stair"


def index_extra(m, rng):
    number = "A7.1"
    m.sheets.append(Sheet(number, "ROOF PLAN", "plan", in_set=False))
    return "G0.1", f"index:{number}", f"The index lists {number}, and the set holds no such sheet.", number


def sheet_unlisted(m, rng):
    sheet = next(s for s in m.sheets if s.number == "A6.2")
    sheet.on_index = False
    return sheet.number, "title", f"{sheet.number} is in the set, and the index does not list it.", sheet.number


def floor_name(m, rng):
    f = _floor(m, 0.2)
    sheet = next(s for s in m.sheets if s.number == _plan(f))
    sheet.block_title = f"{m.floors[f.number].name} PLAN"
    return sheet.number, "title", f"The title block says {sheet.block_title}, and the label under the drawing says {sheet.title}.", "Floor name"


def stories(m, rng):
    declared = len(m.floors) + 1
    m.code["stories"] = str(declared)
    return "G0.1", "code:STORIES", f"The code summary declares {declared} stories, and the set has {len(m.floors)} floor plans.", "Stories declared"


def edition(m, rng):
    m.code["edition"] = "2015 INTERNATIONAL BUILDING CODE"
    return "G0.1", "code:CODE EDITION", "The code summary names the 2015 edition, older than the 2021 code all three cities adopt.", "Code edition"


def area_total(m, rng):
    true = m.load_total["area"]
    m.load_total["area"] = true + 1200
    return "G0.1", "load_total", f"The area total reads {true + 1200:,} SF, and the floors sum to {true:,} SF.", "Area total"


def load_total(m, rng):
    true = m.load_total["load"]
    m.load_total["load"] = true - 9
    return "G0.1", "load_total", f"The occupant total reads {true - 9}, and the floors sum to {true}. A load rounded down is unsafe.", "Occupant total"


def tag_no_row(m, rng):
    f = _floor(m, 0.2)
    mark = _interior(rng, f).mark
    m.door_rows[f.number] = [r for r in m.door_rows[f.number] if r["mark"] != mark]
    return _plan(f), f"tag:{mark}", f"Door tag {mark} has no row in the door schedule.", mark


def row_no_tag(m, rng):
    f = _floor(m, 0.4)
    door = _interior(rng, f)
    door.tag = None
    return "A6.1", f"row:{door.mark}", f"Door schedule row {door.mark} has no tag on {_plan(f)}.", door.mark


def door_type(m, rng):
    f = _floor(m, 0.6)
    door = _interior(rng, f)
    row = next(r for r in m.door_rows[f.number] if r["mark"] == door.mark)
    row["swing"] = "POCKET"
    return _plan(f), f"tag:{door.mark}", f"Door {door.mark} is a pocket door in the schedule, and the plan draws a swing arc.", door.mark


def exit_capacity(m, rng):
    f = _floor(m, 0.4)
    marker = m.markers[f.number][0]
    load = m.load_total["load"]
    marker["capacity"] = math.ceil(load * 0.7)
    marker["width"] = round(marker["capacity"] * 0.3)
    return f"G1.{f.number}", "marker:0", f"The stair marker gives {marker['capacity']} people, below the printed load of {load}.", "Exit capacity"


def room_name(m, rng):
    f = _floor(m, 0.2)
    room = rng.choice(sorted((r for r in f.rooms if r.name == "BEDROOM"), key=lambda r: r.number))
    name = room.name
    room.name = OFFICE
    return _plan(f), f"room:{room.number}", f"Room {room.number} is {OFFICE} on the plan and {name} in the finish schedule.", f"Room {room.number}"


def corridor(m, rng):
    """A wall face moved in with its dimension left behind, so the string still agrees with its own ticks and the sheet keeps its scale."""
    f = _floor(m, 0.8)
    if not f.hall_width:
        raise NotPlantable(f"{f.name} has no hall the corridor check can measure")
    a, b = (np.array(p) for p in f.hall_width)
    printed = float(np.hypot(*(b - a))) * IN_PER_M
    if printed > MAX_PRINTED_IN:
        raise NotPlantable(f"The hall reads {printed:.0f} in, and check 14 compares printed widths of 80 in at most")
    across = (b - a) / float(np.hypot(*(b - a)))
    along = np.array([-across[1], across[0]])
    deep = NARROWED_IN / IN_PER_M
    label = np.array(f.hall.label)
    face, inward = (a, across) if np.dot(label - a, across) > np.dot(b - label, across) else (b, -across)
    mid = face + along * float(np.dot(label - face, along))   # at the label, where the reader measures, because the string stands off it
    run = along * STUB_M
    f.plan_walls.append(Polygon([mid + run, mid - run, mid - run + inward * deep, mid + run + inward * deep]))
    return _plan(f), "hall", f"The hall on {f.name} is dimensioned {printed:.0f} in, and the walls stand {printed - NARROWED_IN:.0f} in apart.", "Corridor on"


def route(m, rng):
    """A route over the limit needs walls that do not exist on a point-access block, so this kind says why rather than bend the building."""
    raise NotPlantable("Check 13 asks at 187.5 ft, 0.75 of the 250 ft limit for sprinklered R-2, and no listed building is long enough: the longest is 27.7 m")


KINDS = {"stair_mid": (stair_mid, 9, ["fix", "check"]), "stair_discharge": (stair_discharge, 10, ["check"]), "index_extra": (index_extra, 2, ["fix"]), "sheet_unlisted": (sheet_unlisted, 2, ["fix"]),
         "floor_name": (floor_name, 3, ["check"]), "stories": (stories, 5, ["fix"]), "edition": (edition, 5, ["fix"]), "area_total": (area_total, 6, ["check"]), "load_total": (load_total, 6, ["check"]),
         "tag_no_row": (tag_no_row, 7, ["fix"]), "row_no_tag": (row_no_tag, 7, ["fix"]), "door_type": (door_type, 7, ["check"]), "exit_capacity": (exit_capacity, 8, ["check"]),
         "room_name": (room_name, 16, ["check"]), "corridor": (corridor, 14, ["check"]), "route": (route, 13, ["check"])}


def plant(m, kind):
    """Edits the model and returns the defect's truth row, with the anchor still to be resolved to a box. The choice depends only on building, seed and kind, so a kind lands in one place in every variant."""
    fn, check, severity = KINDS[kind]
    sheet, anchor, wording, phrase = fn(m, random.Random(f"{m.building}:{m.seed}:{kind}"))
    return {"kind": kind, "check": check, "severity": severity, "sheet": sheet, "anchor": anchor, "wording": wording, "phrase": phrase}
