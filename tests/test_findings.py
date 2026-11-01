"""Findings from small hand-built stage dicts. The base set is consistent and raises no finding, so each test changes one fact."""
import json
import re

import pytest

from storey import findings, floors, meta
from storey.findings import SEVERITY, assign_ids, evaluate
from storey.letter import ASK, draft
from storey.rules import JURISDICTIONS
from storey.verticals import SHAFTS

ORDER = [2, 3, 4]
STATUS = re.compile(r"(?i)\bpass(?:es|ed|ing)?\b|approv|compliant|compliance")
SCAN = "Scanned sheet, so nothing can be measured from it."
NO_ENDS = "No exit markers were read and no stair was found on this floor, so a route has no end."


def cell(stair_id, floor, state, aligned=True, spark="3/3", crop="yes", point=(1000.0, 800.0)):
    return {"state": state, "signals": {"spark": spark, "crop": crop, "ink": 0.9, "aligned": aligned, "reason": f"{state} here"}, "point": list(point), "box": None,
            "crop": f"verticals/{stair_id}_{floor}.jpg", "mask": None}


def stair(stair_id, *states, pos=(1000.0, 800.0), established=True, order=ORDER, name="Stair"):
    return {"id": stair_id, "name": name, "established": established, "pos": list(pos), "floors": {str(f): cell(stair_id, f, s) for f, s in zip(order, states)}}


def sheet(index, number, title, cls="floor_plan", level=""):
    return {"index": index, "number": number, "title": title, "cls": cls, "level": level, "scale": "1/4 in", "plan_box": [0, 0, 6000, 4000], "read_ok": True}


def code_field(value, **extra):
    return {"value": value, "sheet": 0, "box": [10, 10, 90, 20], "reads": 2} | extra


def schedule(sheet_index, kind, rows, plans, source="text"):
    return {"sheet": sheet_index, "kind": kind, "rows": rows, "source": source, "plans": plans}


def row(mark, **cells):
    return {"mark": mark, "cells": cells, "box": [100, 100, 900, 120]}


def tag(sheet_index, mark, kind="door", point=(3000, 2000)):
    return {"sheet": sheet_index, "kind": kind, "mark": mark, "point": list(point)}


def room(number, name, source="text", sheet_index=2):
    return {"sheet": sheet_index, "number": number, "name": name, "source": source, "box": [500, 500, 700, 560]}


def title(index, number, text, under=None, judge=None):
    return {"sheet": index, "number": number, "title": text, "under_drawing": under, "source": "text", "judge": judge, "box": [5000, 3000, 5900, 3900]}


def blank(reason):
    return {"scale_in_per_pt": None, "scale_checked": 0, "doors": [], "routes": [], "corridors": [], "skipped": reason}


def route(length_ft):
    return {"path": [[100, 100], [100, 900]], "length_ft": length_ft, "from_room": "Office", "to_exit": "Stair A", "end_source": "matrix", "label": "Minimum. Examiner to confirm."}


def measured(length_ft=100.0):
    return {"scale_in_per_pt": 0.667, "scale_checked": 12, "doors": [], "routes": [route(length_ft)], "corridors": [hall(100.0, 106.0)], "skipped": None}


def base():
    sheets = [sheet(0, "G0.1", "CODE SUMMARY", "code_summary"), sheet(1, "A1.1", "LIFE SAFETY PLAN", level="FIRST FLOOR"), sheet(2, "A2.1", "FIRST FLOOR PLAN", level="FIRST FLOOR"),
              sheet(3, "A2.2", "SECOND FLOOR PLAN", level="SECOND FLOOR"), sheet(4, "A2.3", "THIRD FLOOR PLAN", level="THIRD FLOOR"), sheet(5, "A6.1", "DOOR SCHEDULE", "schedule")]
    code = {"edition": code_field("2021 International Building Code"), "occupancy": code_field("B", mentions=[]), "construction": code_field("IIB"), "sprinklers": code_field("yes", mentions=[]),
            "stories": code_field("3 STORIES"), "area": code_field("30,000 SF"), "occupant_load": code_field("120")}
    return {
        "read": {"sheets": sheets},
        "floors": {"order": list(ORDER), "reference": 2, "transforms": {}, "inliers": {}, "unaligned": [], "stack": "stack.mp4", "frame_size": [960, 640], "discharge": 2, "not_checked": None},
        "verticals": {"verticals": [stair("A", "found", "found", "found"), stair("B", "found", "found", "found", pos=(4000.0, 800.0))], "not_checked": [SHAFTS]},
        "declared": {
            "code": code,
            "index": [{"number": s["number"], "title": s["title"], "not_used": False, "sheet": 0, "box": [10, 10 * i, 90, 10 * i + 8]} for i, s in enumerate(sheets)],
            "titles": [title(s["index"], s["number"], s["title"]) for s in sheets],
            "tags": [tag(2, "1"), tag(2, "2", point=(3400, 2000))],
            "schedules": [schedule(5, "door", [row("1", SWING="LEFT", SIZE='36" x 80"'), row("2", SWING="RIGHT", SIZE='36" x 80"')], [2]), schedule(5, "finish", [row("101", ROOM="OFFICE")], [])],
            "tables": [],
            "rooms": [room("101", "Office")],
            "exits": {"sheets": []},
            "cannot_tell": [],
        },
        "vector": {"sheets": {str(i): measured() if i in ORDER else blank("Not a floor plan.") for i in range(6)}},
    }


def run(stages, jurisdiction="houston", sprinklers=None, discharge=None):
    out = evaluate(stages, jurisdiction, sprinklers, discharge)
    assert not [line for line in out["not_checked"] if "check did not finish" in line]
    return out


def raised(stages, check, **options):
    return [f for f in run(stages, **options)["findings"] if f["check"] == check]


def only(stages, check, **options):
    items = raised(stages, check, **options)
    assert len(items) == 1, items
    return items[0]


def opens(stages, **options):
    return run(stages, **options)["open_items"]


def strings(value):
    """Every key and string in a nested value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def unread(stages, index):
    stages["read"]["sheets"][index] |= {"read_ok": False, "number": "", "title": "", "cls": "other", "level": "", "scale": ""}
    stages["vector"]["sheets"][str(index)] = blank("The sheet read has no result for this sheet.")


def with_stairs(stages, *rows):
    stages["verticals"]["verticals"] = list(rows)
    return stages


def storeys(stages, levels):
    """Replaces the plans with one sheet per level, starting at sheet 10."""
    order = list(range(10, 10 + len(levels)))
    stages["read"]["sheets"] = [r for r in stages["read"]["sheets"] if r["index"] not in ORDER] + [sheet(i, f"A3.{i}", f"{level} PLAN", level=level) for i, level in zip(order, levels)]
    stages["floors"]["order"] = order
    stages["vector"]["sheets"] |= {str(i): measured() for i in order}
    return order


def test_the_base_set_raises_nothing():
    out = run(base())
    assert out["findings"] == []
    assert [i["check"] for i in out["open_items"]] == [12, 12]


def test_severity_table_matches_the_spec():
    assert SEVERITY == {1: "fix", 2: "fix", 3: "check", 4: "check", 5: "fix", 6: "check", 7: "fix", 8: "unknown", 9: "fix", 10: "fix", 11: "check", 12: "open", 13: "check", 14: "check", 16: "check"}


def test_finding_has_every_field():
    s = base()
    s["declared"]["tags"].append(tag(2, "9"))
    f = only(s, 7)
    assert set(f) == {"key", "check", "severity", "discipline", "sheet", "box", "title", "measured", "required", "rule", "evidence", "vertical"}
    assert f["discipline"] == "A"


def test_every_check_has_a_letter_instruction_and_output_drafts():
    s = base()
    s["declared"]["tags"].append(tag(2, "9"))
    out = run(s)
    letter = draft(out["findings"], {"sheets": s["read"]["sheets"], "not_checked": out["not_checked"]})
    assert letter["items"][0]["sheet"] == "A2.1" and letter["items"][0]["text"].startswith(ASK[7])
    assert set(SEVERITY) - {12} <= set(ASK)


# Check 1


def test_check_1_absent_sheets_are_fix():
    s = base()
    s["read"]["sheets"] = [r for r in s["read"]["sheets"] if r["index"] in (2, 3, 4)]
    s["declared"] |= {"code": {}, "schedules": [], "titles": [t for t in s["declared"]["titles"] if t["sheet"] in (2, 3, 4)]}
    items = raised(s, 1)
    assert {f["severity"] for f in items} == {"fix"} and all(f["sheet"] is None for f in items)
    assert len(items) == 3
    assert all(any(name in f["title"] for f in items) for name in ("code summary", "life safety", "door schedule"))


def test_check_1_no_floor_plan_is_fix():
    s = base()
    for r in s["read"]["sheets"]:
        r["cls"] = "other"
    assert "floor plan" in only(s, 1)["title"]


def test_check_1_cover_with_code_data_counts_as_a_code_summary():
    s = base()
    s["read"]["sheets"][0] |= {"cls": "cover", "title": "COVER SHEET"}
    s["declared"]["titles"][0]["title"] = "COVER SHEET"
    assert raised(s, 1) == []


def test_check_1_unread_sheet_makes_an_absent_sheet_unknown():
    s = base()
    s["read"]["sheets"][5] |= {"read_ok": False, "number": "", "title": "", "cls": "other"}
    s["declared"] |= {"schedules": [], "titles": [t for t in s["declared"]["titles"] if t["sheet"] != 5]}
    assert [f["severity"] for f in raised(s, 1) if "door schedule" in f["title"]] == ["unknown"]


def test_check_1_door_schedule_named_on_a_sheet_is_not_absent():
    s = base()
    s["read"]["sheets"][5]["title"] = s["declared"]["titles"][5]["title"] = "SCHEDULES"
    s["declared"]["index"][5]["title"] = "SCHEDULES"
    s["declared"]["schedules"] = []
    s["declared"]["cannot_tell"] = [{"what": "Door schedule", "why": "No ruled table was found on this sheet, so it was not read.", "sheet": 5}]
    assert [f for f in raised(s, 1) if f["severity"] == "fix"] == []


def test_check_1_door_schedule_found_by_its_title_alone():
    s = base()
    s["declared"]["schedules"] = []
    assert raised(s, 1) == []


def test_check_1_life_safety_plan_found_by_its_exit_markers_alone():
    s = base()
    s["read"]["sheets"][1] |= {"title": "FIRST FLOOR LAYOUT", "cls": "detail"}
    s["declared"]["titles"][1]["title"] = "FIRST FLOOR LAYOUT"
    assert [f["title"] for f in raised(s, 1)] == ["No life safety plan in the set"]
    s["declared"]["exits"]["sheets"] = [exit_sheet(200)]
    assert raised(s, 1) == []


def test_check_1_code_summary_found_by_its_class_alone():
    s = base()
    s["read"]["sheets"][0]["title"] = s["declared"]["titles"][0]["title"] = "GENERAL NOTES"
    s["declared"]["code"] = {}
    assert raised(s, 1) == []


# Check 2


def test_check_2_sheet_not_on_the_index():
    s = base()
    s["declared"]["index"].pop()
    f = only(s, 2)
    assert f["severity"] == "fix" and f["sheet"] == 5 and "A6.1" in f["title"]


def test_check_2_index_row_with_no_sheet():
    s = base()
    s["declared"]["index"].append({"number": "A9.9", "title": "DETAILS", "not_used": False, "sheet": 0, "box": [1, 2, 3, 4]})
    f = only(s, 2)
    assert f["severity"] == "fix" and f["sheet"] == 0 and f["box"] == [1, 2, 3, 4] and "A9.9" in f["title"]


def test_check_2_not_used_row_is_skipped():
    s = base()
    s["declared"]["index"].append({"number": "A9.9", "title": "NOT USED", "not_used": True, "sheet": 0, "box": None})
    assert raised(s, 2) == []


def test_check_2_sheet_listed_only_as_not_used_is_on_the_index():
    s = base()
    s["declared"]["index"][5]["not_used"] = True
    assert raised(s, 2) == []


def test_check_2_number_listed_twice_is_check():
    s = base()
    s["declared"]["index"].append(dict(s["declared"]["index"][0]))
    assert only(s, 2)["severity"] == "check"


def test_check_2_two_sheets_with_one_number_is_check():
    s = base()
    s["declared"]["titles"][5]["number"] = "A2.3"
    items = raised(s, 2)
    assert {f["severity"] for f in items} == {"check", "fix"}   # the duplicate, and index row A6.1 left with no sheet


def test_check_2_no_index_raises_nothing_but_the_read_note():
    s = base()
    s["declared"]["index"] = []
    s["declared"]["cannot_tell"] = [{"what": "Sheet index", "why": "No sheet index was read in this set.", "sheet": None}]
    assert only(s, 2)["severity"] == "unknown"


def test_check_2_blank_sheet_number_makes_an_orphan_row_unknown():
    s = base()
    s["declared"]["titles"][5] |= {"number": "", "title": ""}
    s["declared"]["cannot_tell"] = [{"what": "Sheet number", "why": "Two reads gave A6.1 and A6.7.", "sheet": 5}]
    assert {f["severity"] for f in raised(s, 2)} == {"unknown"}


def test_check_2_partial_index_makes_an_unlisted_sheet_unknown():
    s = base()
    s["declared"]["index"].pop()
    s["declared"]["cannot_tell"] = [{"what": "Sheet index", "why": "1 of 8 crops of this sheet could not be read, so entries can be missing.", "sheet": 0}]
    assert {f["severity"] for f in raised(s, 2)} == {"unknown"}


# Check 3


def test_floor_word_conflict_is_flagged():
    s = base()
    s["declared"]["titles"][2] = title(2, "A2.1", "MAIN FLOOR PLAN", under="Ground Floor Plan; GROUND LEVEL DOOR SCHEDULE")
    f = only(s, 3)
    assert f["severity"] == "check" and f["sheet"] == 2
    assert "MAIN FLOOR PLAN" in f["measured"] and "Ground Floor Plan" in f["measured"]


def test_floor_conflict_with_the_index_is_flagged_though_one_read_said_same():
    s = base()
    s["declared"]["titles"][2] = title(2, "A2.1", "MAIN FLOOR PLAN", judge="same")
    s["declared"]["index"][2]["title"] = "Ground Floor Plan"
    assert only(s, 3)["sheet"] == 2


def test_block_conflict_with_the_index_is_flagged():
    s = base()
    s["declared"]["titles"][2] = title(2, "A2.1", "FIRST FLOOR PLAN - BLOCK 2C", judge="same")
    s["declared"]["index"][2]["title"] = "FIRST FLOOR PLAN - BLOCK 1C"
    assert only(s, 3)["sheet"] == 2


def test_floor_conflict_on_one_sheet_is_one_finding():
    s = base()
    s["declared"]["titles"][2] = title(2, "A2.1", "MAIN FLOOR PLAN", under="Ground Floor Plan", judge="same")
    s["declared"]["index"][2]["title"] = "Ground Floor Plan"
    f = only(s, 3)
    assert "index" in f["measured"].lower() and "drawing" in f["measured"].lower()


def test_sheet_that_matches_one_of_two_index_rows_is_not_a_title_difference():
    s = base()
    s["declared"]["index"].append(s["declared"]["index"][2] | {"title": "SECOND FLOOR PLAN"})
    assert raised(s, 3) == []


def test_title_judged_different_in_both_reads_is_check():
    s = base()
    s["declared"]["titles"][5] = title(5, "A6.1", "PLUMBING SCHEDULES", judge="different")
    s["declared"]["index"][5]["title"] = "HVAC Cover Sheet"
    f = only(s, 3)
    assert f["severity"] == "check" and "HVAC Cover Sheet" in f["measured"]


# Check 4


@pytest.mark.parametrize("field", ["edition", "occupancy", "construction", "sprinklers", "stories", "area", "occupant_load"])
def test_check_4_field_with_nothing_printed_is_check(field):
    s = base()
    s["declared"]["code"][field]["value"] = None
    assert [f["severity"] for f in raised(s, 4) if "has no" in f["title"]] == ["check"]


def test_check_4_edition_with_no_year_is_check():
    s = base()
    s["declared"]["code"]["edition"]["value"] = "International Building Code"
    assert "year" in only(s, 4)["title"]


@pytest.mark.parametrize("value, mention", [("no", "SPRINKLER HEADS AT CEILING"), ("yes", "THIS BUILDING IS NOT SPRINKLERED")])
def test_check_4_sprinkler_statements_that_disagree_are_check(value, mention):
    s = base()
    s["declared"]["code"]["sprinklers"] |= {"value": value, "mentions": [mention]}
    assert only(s, 4)["severity"] == "check"


def test_check_4_field_two_reads_disagreed_on_is_unknown_once():
    s = base()
    s["declared"]["code"]["area"] |= {"value": None, "reads": 0}
    s["declared"]["cannot_tell"] = [{"what": "Code field area", "why": "Two reads gave 3,000 SF and 8,000 SF.", "sheet": 0}]
    assert only(s, 4)["severity"] == "unknown"


# Check 5


def test_check_5_stories_against_plans_is_fix():
    s = base()
    s["declared"]["code"]["stories"]["value"] = "5 STORIES"
    f = only(s, 5)
    assert f["severity"] == "fix" and "5" in f["measured"] and "3" in f["measured"]


@pytest.mark.parametrize("value", ["3", "THREE STORIES", "3 STORIES + BASEMENT"])
def test_check_5_stories_read_in_any_form(value):
    s = base()
    s["declared"]["code"]["stories"]["value"] = value
    assert raised(s, 5) == []


@pytest.mark.parametrize("value", ["3", "4"])
def test_check_5_basement_and_roof_plans_do_not_miscount(value):
    s = base()
    storeys(s, ["BASEMENT", "FIRST FLOOR", "SECOND FLOOR", "THIRD FLOOR", "ROOF", "FOUNDATION"])
    s["declared"]["code"]["stories"]["value"] = value
    assert raised(s, 5) == []


def test_check_5_unnamed_plan_makes_stories_unknown():
    s = base()
    s["read"]["sheets"].append(sheet(6, "A2.4", "PLAN", level=""))
    s["declared"]["code"]["stories"]["value"] = "4"
    assert only(s, 5)["severity"] == "unknown"


def test_check_5_stories_in_words_are_compared():
    s = base()
    s["declared"]["code"]["stories"]["value"] = "FIVE STORIES"
    f = only(s, 5)
    assert f["severity"] == "fix" and "5" in f["measured"]


def test_check_5_unnamed_roof_plan_leaves_the_fix():
    s = base()
    s["read"]["sheets"].append(sheet(6, "A2.4", "ROOF PLAN", level=""))
    s["declared"]["code"]["stories"]["value"] = "4"
    assert only(s, 5)["severity"] == "fix"


def test_check_5_sheet_dropped_as_a_repeat_makes_stories_unknown():
    s = base()
    s["read"]["sheets"][3]["level"] = "FIRST FLOOR"   # bme_k's first run read sheets 2 and 3 both as I. EMELET.
    s["floors"]["order"] = [2, 4]
    f = only(s, 5)
    assert f["severity"] == "unknown"


def test_check_5_unread_sheet_makes_stories_unknown():
    s = base()
    unread(s, 3)
    s["floors"]["order"] = [2, 4]
    assert only(s, 5)["severity"] == "unknown"


def test_check_5_edition_against_the_adopted_edition_is_fix():
    s = base()
    s["declared"]["code"]["edition"]["value"] = "2018 International Building Code"
    f = only(s, 5)
    assert f["severity"] == "fix" and f["sheet"] == 0 and f["required"] == JURISDICTIONS["houston"]


# Check 6


def occupant_table(*rows):
    return {"sheet": 0, "kind": "occupant", "rows": list(rows), "box": [0, 0, 900, 900]}


def load_row(area, factor, occupants, number="101"):
    return {"room": number, "name": "OFFICE", "area": area, "factor": factor, "occupants": occupants, "box": [0, 10, 900, 20]}


def total_row(area, occupants):
    return {"room": None, "name": "Grand total", "area": area, "factor": None, "occupants": occupants, "total": True, "box": [0, 800, 900, 820]}


def exit_row(name, clear, occupants, required):
    return {"exit": name, "clear_in": clear, "occupants": occupants, "required_in": required, "box": [0, 10, 900, 20]}


def test_check_6_consistent_tables_raise_nothing():
    s = base()
    s["declared"]["tables"] = [occupant_table(load_row(3000.0, 150.0, 20), load_row(4351.0, 100.0, 44, "102"), total_row(7351.0, 64)),
                               {"sheet": 0, "kind": "exit", "rows": [exit_row("EXIT 1", 36.0, 40, 8.0), exit_row("EXIT 2", 36.0, 24, 7.2), {"exit": "EXIT 1", "occupants": 40, "required_in": 8.0, "example": True, "box": [0, 0, 1, 1]}], "box": [0, 0, 9, 9]},
                               {"sheet": 0, "kind": "area", "rows": [{"label": "Main Level", "area": 576.0, "box": [0, 0, 1, 1]}, {"label": "Upper Level", "area": 576.0, "box": [0, 2, 1, 3]}, {"label": "total", "area": 1152.0, "total": True, "box": [0, 4, 1, 5]}], "box": [0, 0, 9, 9]}]
    assert raised(s, 6) == []


def test_check_6_load_rounded_down_says_unsafe():
    s = base()
    s["declared"]["tables"] = [occupant_table(load_row(4351.0, 100.0, 43))]
    f = only(s, 6)
    assert f["severity"] == "check" and "unsafe" in f["measured"] and "44" in f["measured"] and "43" in f["measured"]


def test_check_6_load_over_declared_says_safe():
    s = base()
    s["declared"]["tables"] = [occupant_table(load_row(150.0, 150.0, 2))]
    f = only(s, 6)
    assert "safe" in f["measured"] and "unsafe" not in f["measured"]


def test_check_6_spark_rows_have_no_room_or_box():
    s = base()
    s["declared"]["tables"] = [occupant_table({"area": 4351.0, "factor": 100.0, "occupants": 43})]
    assert "unsafe" in only(s, 6)["measured"]


def test_check_6_occupant_total_against_rows():
    s = base()
    s["declared"]["tables"] = [occupant_table(load_row(3000.0, 150.0, 20), total_row(3000.0, 25))]
    assert "total" in only(s, 6)["title"].lower()


def test_check_6_area_total_has_a_rounding_tolerance():
    s = base()
    s["declared"]["tables"] = [occupant_table(load_row(11542.0, None, None), total_row(11538.0, 0))]
    assert raised(s, 6) == []
    s["declared"]["tables"] = [occupant_table(load_row(12000.0, None, None), total_row(11538.0, 0))]
    assert "area" in only(s, 6)["title"].lower()


def test_check_6_exit_width_against_occupants():
    s = base()
    s["declared"]["tables"] = [{"sheet": 0, "kind": "exit", "rows": [exit_row("EXIT 1", 36.0, 90, 15.0)], "box": [0, 0, 9, 9]}]
    f = only(s, 6)
    assert "18" in f["measured"] and "unsafe" in f["measured"] and "1005.3.2" in f["rule"]


def test_check_6_width_above_the_worked_figure_says_safe():
    s = base()
    s["declared"]["tables"] = [{"sheet": 0, "kind": "exit", "rows": [exit_row("EXIT 1", 36.0, 100, 25.0)], "box": [0, 0, 9, 9]}]
    f = only(s, 6)
    assert "safe" in f["measured"] and "unsafe" not in f["measured"]


def width_at_the_voice_alarm_factor(s):
    s["declared"]["tables"] = [{"sheet": 0, "kind": "exit", "rows": [exit_row("EXIT 1", 36.0, 100, 15.0)], "box": [0, 0, 9, 9]}]
    return s


def test_check_6_voice_alarm_factor_on_a_sprinklered_building_is_a_value_to_confirm():
    f = only(width_at_the_voice_alarm_factor(base()), 6)
    assert f["severity"] == "check" and "voice alarm" in f["measured"] and "unsafe" not in f["measured"]


def test_check_6_voice_alarm_factor_with_no_sprinklers_is_unsafe():
    f = only(width_at_the_voice_alarm_factor(base()), 6, sprinklers="no")
    assert "unsafe" in f["measured"] and "voice alarm" not in f["measured"]


def test_check_6_voice_alarm_factor_with_sprinklers_unknown_is_unsafe():
    s = width_at_the_voice_alarm_factor(base())
    s["declared"]["code"]["sprinklers"]["value"] = None
    assert "unsafe" in only(s, 6)["measured"]


@pytest.mark.parametrize("occupancy", ["I-2", None])
def test_check_6_voice_alarm_factor_with_no_group_in_its_row_is_unsafe(occupancy):
    s = width_at_the_voice_alarm_factor(base())
    s["declared"]["code"]["occupancy"]["value"] = occupancy
    assert "unsafe" in only(s, 6)["measured"]


def test_check_6_mixed_occupancy_can_use_the_voice_alarm_factor():
    s = width_at_the_voice_alarm_factor(base())
    s["declared"]["code"]["occupancy"]["value"] = "B, I-2"
    assert "voice alarm" in only(s, 6)["measured"]


def test_check_6_stair_factor_is_arithmetic_that_holds():
    s = base()
    s["declared"]["tables"] = [{"sheet": 0, "kind": "exit", "rows": [exit_row("EXIT 1", 36.0, 90, 27.0)], "box": [0, 0, 9, 9]}]
    assert raised(s, 6) == []


def test_check_6_clear_width_below_required():
    s = base()
    s["declared"]["tables"] = [{"sheet": 0, "kind": "exit", "rows": [exit_row("EXIT 1", 16.0, 90, 18.0)], "box": [0, 0, 9, 9]}]
    assert "EXIT 1" in only(s, 6)["title"]


def test_check_6_worked_example_against_the_table():
    s = base()
    s["declared"]["tables"] = [{"sheet": 0, "kind": "exit", "rows": [exit_row("EXIT 1", 33.0, 90, 18.0), {"exit": "EXIT 1", "occupants": 100, "required_in": 20.0, "example": True, "box": [0, 40, 9, 49]}], "box": [0, 0, 9, 9]}]
    f = only(s, 6)
    assert "100" in f["measured"] and "90" in f["measured"]


def test_check_6_worked_example_finds_its_corridor_by_number():
    s = base()
    s["declared"]["tables"] = [{"sheet": 0, "kind": "exit", "rows": [exit_row("140 SECURE CIRCULATION", 60.0, 68, 13.6), {"exit": "CORRIDOR 140", "occupants": 70, "required_in": 14.0, "example": True, "box": [0, 40, 9, 49]}], "box": [0, 0, 9, 9]}]
    assert "70" in only(s, 6)["measured"]


def test_check_6_worked_example_with_no_name_is_left_alone():
    s = base()
    s["declared"]["tables"] = [{"sheet": 0, "kind": "exit", "rows": [exit_row("EXIT 1", 33.0, 90, 18.0), {"exit": "", "occupants": 100, "required_in": 20.0, "example": True, "box": [0, 40, 9, 49]}], "box": [0, 0, 9, 9]}]
    assert raised(s, 6) == []


def test_check_6_exit_sum_against_the_occupant_total():
    s = base()
    s["declared"]["tables"] = [occupant_table(load_row(3000.0, 150.0, 20), total_row(3000.0, 20)),
                               {"sheet": 0, "kind": "exit", "rows": [exit_row("EXIT 1", 33.0, 15, 3.0), exit_row("EXIT 2", 33.0, 15, 3.0)], "box": [0, 0, 9, 9]}]
    f = only(s, 6)
    assert "30" in f["measured"] and "20" in f["measured"] and "safe" in f["measured"]


def test_check_6_net_area_note_against_rows_with_no_load():
    s = base()
    s["declared"]["tables"] = [occupant_table(load_row(9075.0, None, None), load_row(2463.0, None, None, "102"), total_row(11538.0, 0),
                                              {"note": "(11,538 SF) - (CIRCULATION + RESTROOMS) = 9,273 SF", "area": 11538.0, "net_area": 9273.0, "box": [0, 900, 900, 920]})]
    assert raised(s, 6) == []   # every row here has no load, so the sum is the whole area and the note cannot be tested
    s["declared"]["tables"][0]["rows"][0] |= {"factor": 150.0, "occupants": 61}
    s["declared"]["tables"][0]["rows"][2]["occupants"] = 61
    f = only(s, 6)
    assert "9,273" in f["measured"] or "9273" in f["measured"]


def test_check_6_unparsed_number_is_left_alone():
    s = base()
    s["declared"]["tables"] = [{"sheet": 0, "kind": "exit", "rows": [exit_row("EXIT 1", None, 90, None)], "box": [0, 0, 9, 9]}, occupant_table(load_row(None, 100.0, 43))]
    assert raised(s, 6) == []


# Check 7


def test_check_7_tag_with_no_row_is_fix_on_the_plan():
    s = base()
    s["declared"]["tags"].append(tag(2, "13"))
    f = only(s, 7)
    assert f["severity"] == "fix" and f["sheet"] == 2 and "13" in f["title"]


def test_check_7_row_with_no_tag_is_fix_on_the_schedule():
    s = base()
    s["declared"]["schedules"][0]["rows"].append(row("9", SWING="LEFT"))
    f = only(s, 7)
    assert f["severity"] == "fix" and f["sheet"] == 5 and "9" in f["title"]


def test_check_7_marks_of_one_schedule_are_one_finding():
    s = base()
    s["declared"]["schedules"][0]["rows"] += [row(str(n), SWING="LEFT") for n in range(3, 17)]
    s["declared"]["tags"] += [tag(2, str(n)) for n in range(19, 35)]
    tags, rows = sorted(raised(s, 7), key=lambda f: f["sheet"])
    assert (tags["sheet"], rows["sheet"]) == (2, 5)
    assert "19, 20" in tags["title"] and "and 6 more" in tags["title"] and "16 of 18" in tags["measured"]
    assert "3, 4" in rows["title"] and "14 of 16" in rows["measured"]


def test_check_7_marks_compare_without_case_or_spaces():
    s = base()
    s["declared"]["schedules"][0]["rows"][0]["mark"] = "1 a"
    s["declared"]["tags"][0]["mark"] = "1A"
    assert raised(s, 7) == []


def test_check_7_two_schedules_on_one_plan_share_their_marks():
    s = base()
    s["declared"]["schedules"].append(schedule(5, "door", [row("7", SWING="LEFT")], [2]))
    s["declared"]["tags"].append(tag(2, "7"))
    assert raised(s, 7) == []


@pytest.mark.parametrize("what", ["Door tags", "Door schedule", "Plan not read", "Text layer", "Spark reads"])
def test_check_7_partial_read_makes_a_missing_partner_unknown(what):
    s = base()
    s["declared"]["tags"].append(tag(2, "13"))
    s["declared"]["cannot_tell"] = [{"what": what, "why": "Only one read saw 13.", "sheet": 2}]
    assert {f["severity"] for f in raised(s, 7)} == {"unknown"}


def test_check_7_partial_read_of_another_sheet_leaves_the_fix():
    s = base()
    s["declared"]["tags"].append(tag(2, "13"))
    s["declared"]["cannot_tell"] = [{"what": "Door tags", "why": "Only one read saw 13.", "sheet": 3}]
    assert "fix" in {f["severity"] for f in raised(s, 7)}


def test_check_7_schedule_with_no_plan_or_tags_compares_nothing():
    s = base()
    s["declared"]["schedules"][0]["plans"] = []
    s["declared"]["tags"] = []
    assert raised(s, 7) == []


def windows(qty, *tags):
    return schedule(2, "window", [row("A", QTY=qty, WIDTH="2'-5\"")], [2]), [tag(2, m, "window") for m in tags]


@pytest.mark.parametrize("qty, marks, severity", [("2", ["A", "A"], None), ("3", ["A", "A"], "check"), ("0", ["A"], "check"), ("2", [], None), ("0", [], None)])
def test_check_7_window_count_against_quantity(qty, marks, severity):
    s = base()
    table, tags = windows(qty, *marks)
    s["declared"]["schedules"].append(table)
    s["declared"]["tags"] += tags
    assert [f["severity"] for f in raised(s, 7)] == ([severity] if severity else [])


def test_check_7_window_row_with_a_quantity_and_no_tag_at_all_is_fix():
    s = base()
    table, tags = windows("2", "B")
    table["rows"].append(row("B", QTY="1"))
    s["declared"]["schedules"].append(table)
    s["declared"]["tags"] += tags
    f = only(s, 7)
    assert f["severity"] == "fix" and "A" in f["title"]


def test_check_7_window_row_counted_as_none_needs_no_tag():
    s = base()
    table, tags = windows("2", "A", "A")
    table["rows"].append(row("B", QTY="0"))
    s["declared"]["schedules"].append(table)
    s["declared"]["tags"] += tags
    assert raised(s, 7) == []


def door(point=(3000, 2000), hinge=(2940, 2000), width_in=36.0):
    return {"point": list(point), "width_in": width_in, "hinge": list(hinge)}


def swing_at_tag_1(swing):
    s = base()
    s["declared"]["schedules"][0]["rows"][0]["cells"]["SWING"] = swing
    s["vector"]["sheets"]["2"]["doors"] = [door()]
    return s


@pytest.mark.parametrize("swing", ["BIFOLD", "POCKET", "bifold"])
def test_check_7_swing_arc_on_a_row_with_no_swing_is_check(swing):
    f = only(swing_at_tag_1(swing), 7)
    assert f["severity"] == "check" and f["sheet"] == 2 and swing in f["measured"] and "36" in f["measured"]


def test_check_7_row_with_no_size_reads_plainly():
    s = swing_at_tag_1("BIFOLD")
    del s["declared"]["schedules"][0]["rows"][0]["cells"]["SIZE"]
    assert only(s, 7)["measured"] == "The schedule says BIFOLD, and the plan shows a swing arc of 36 in."


@pytest.mark.parametrize("swing", ["LEFT", "RIGHT", "-", ""])
def test_check_7_arc_on_any_other_row_raises_nothing(swing):
    assert raised(swing_at_tag_1(swing), 7) == []   # a dash is a cased opening, and the door list keeps the dashed swings drawn as options


def test_check_7_arc_past_the_reach_of_a_tag_is_not_its_door():
    s = swing_at_tag_1("BIFOLD")
    s["vector"]["sheets"]["2"]["doors"] = [door(point=(3180, 2000), hinge=(3120, 2000))]   # 54 drawing inches from tag 1, the far side of the reach
    assert raised(s, 7) == []


def test_check_7_a_tag_keeps_its_nearest_arc():
    s = swing_at_tag_1("BIFOLD")
    s["vector"]["sheets"]["2"]["doors"] = [door(), door(point=(3030, 2000), hinge=(2970, 2000), width_in=32.0)]
    f = only(s, 7)
    assert "36" in f["measured"].split("arc of")[1]


def test_check_7_arcs_need_the_vector_stage_only():
    s = swing_at_tag_1("BIFOLD")
    s["vector"] = {"failed": "Measuring did not finish."}
    out = run(s)
    assert [f for f in out["findings"] if f["check"] == 7] == []
    assert any("did not finish" in line for line in out["not_checked"])


# Check 8


def exit_sheet(total, reads=None):
    return {"sheet": 1, "sum": total, "reads": reads or [total, total], "markers": [{"point": [10, 10], "width_mm": 915.0, "capacity": total}]}


def test_exit_sum_never_reads_as_pass():
    s = base()
    s["declared"]["exits"]["sheets"] = [exit_sheet(1440)]
    f = only(s, 8)
    assert f["severity"] == "unknown" and "1440" in f["measured"]


def test_check_8_sum_below_the_printed_load_is_check():
    s = base()
    s["declared"]["exits"]["sheets"] = [exit_sheet(100)]
    f = only(s, 8)
    assert f["severity"] == "check" and "100" in f["measured"] and "120" in f["required"]


def test_check_8_with_no_printed_load_stays_unknown():
    s = base()
    s["declared"]["exits"]["sheets"] = [exit_sheet(100)]
    s["declared"]["code"]["occupant_load"]["value"] = None
    assert [f["severity"] for f in raised(s, 8)] == ["unknown"]


def test_check_8_two_reads_that_disagree_give_only_the_read_note():
    s = base()
    s["declared"]["exits"]["sheets"] = [{"sheet": 1, "sum": None, "reads": [100, 90], "markers": []}]
    s["declared"]["cannot_tell"] = [{"what": "Exit capacity", "why": "Two reads summed to 100 and 90 people.", "sheet": 1}]
    assert only(s, 8)["severity"] == "unknown"


# Checks 9 and 10


def test_missing_cell_between_found_is_fix():
    s = with_stairs(base(), stair("A", "found", "missing", "found"))
    f = only(s, 9)
    assert f["severity"] == "fix" and f["sheet"] == 3 and f["vertical"] == "A"
    assert "FIRST FLOOR" in f["measured"] and "THIRD FLOOR" in f["measured"]


def test_check_cell_between_found_is_check():
    assert only(with_stairs(base(), stair("A", "found", "check", "found")), 9)["severity"] == "check"


def test_found_floors_need_not_be_next_to_the_gap():
    s = base()
    order = storeys(s, ["FIRST FLOOR", "SECOND FLOOR", "THIRD FLOOR", "FOURTH FLOOR"])
    s["floors"]["discharge"] = order[0]
    with_stairs(s, stair("A", "found", "check", "missing", "found", order=order))
    assert sorted(f["severity"] for f in raised(s, 9)) == ["check", "fix"]


def test_gap_beside_a_check_floor_gives_no_check_9_and_stays_visible():
    s = base()
    order = storeys(s, ["FIRST FLOOR", "SECOND FLOOR", "THIRD FLOOR", "FOURTH FLOOR", "FIFTH FLOOR"])
    s["floors"]["discharge"] = order[0]
    with_stairs(s, stair("A", "found", "missing", "check", "end", "end", order=order))
    assert raised(s, 9) == []
    assert sum(item["sheet"] == order[1] for item in opens(s)) == 1


def test_end_cells_are_not_gaps():
    assert raised(with_stairs(base(), stair("A", "found", "found", "end")), 9) == []


def test_row_seen_by_one_run_raises_no_gap():
    assert raised(with_stairs(base(), stair("A", "found", "missing", "found", established=False)), 9) == []


def test_unaligned_floor_never_gives_a_fix():
    s = with_stairs(base(), stair("A", "found", "missing", "found"))
    s["verticals"]["verticals"][0]["floors"]["3"]["signals"]["aligned"] = False
    assert only(s, 9)["severity"] == "check"
    s = with_stairs(base(), stair("A", "found", "missing", "found"))
    s["floors"]["unaligned"] = [3]
    assert only(s, 9)["severity"] == "check"


def test_unaligned_discharge_floor_gives_no_fix():
    s = with_stairs(base(), stair("A", "end", "found", "found"))
    s["verticals"]["verticals"][0]["floors"]["2"]["signals"]["aligned"] = False
    assert only(s, 10)["severity"] == "check"


def test_stair_stopping_above_discharge_raises_check_10():
    f = only(with_stairs(base(), stair("A", "end", "found", "found")), 10)
    assert f["severity"] == "fix" and f["sheet"] == 2 and f["vertical"] == "A" and "1023.3" in f["rule"]


@pytest.mark.parametrize("state, word", [("end", "Not on this floor"), ("missing", "Missing"), ("check", "Check")])
def test_discharge_item_uses_the_words_on_the_screen(state, word):
    assert only(with_stairs(base(), stair("A", state, "found", "found")), 10)["measured"].endswith(f"it reads {word}.")


def test_missing_on_discharge_with_found_above_raises_check_10():
    assert only(with_stairs(base(), stair("A", "missing", "found", "found")), 10)["severity"] == "fix"


def test_check_cell_on_discharge_with_found_above_is_check():
    assert only(with_stairs(base(), stair("A", "check", "found", "found")), 10)["severity"] == "check"


def test_stair_below_discharge_only_raises_nothing():
    s = with_stairs(base(), stair("A", "found", "end", "end"))
    s["floors"]["discharge"] = 3
    assert raised(s, 10) == []


ORDER_LOST = "The floor order could not be read, so floors are in sheet order. Check the order before relying on stair continuity."


def test_gap_with_the_floor_order_not_read_is_check():
    s = base()
    order = storeys(s, ["BASEMENT", "FOURTH FLOOR", "FIRST FLOOR", "SECOND FLOOR", "THIRD FLOOR"])   # habs in sheet order, with a stair that really stops at the third floor
    s["floors"] |= {"discharge": order[2], "not_checked": f"No floor name was read on A2.9, so stairs were not followed there. {ORDER_LOST}"}
    with_stairs(s, stair("A", "found", "missing", "found", "found", "found", order=order))
    f = only(s, 9)
    assert f["severity"] == "check" and "sheet order" in f["measured"]


def test_stair_above_discharge_with_the_floor_order_not_read_is_check():
    s = with_stairs(base(), stair("A", "end", "found", "found"))
    s["floors"]["not_checked"] = ORDER_LOST
    f = only(s, 10)
    assert f["severity"] == "check" and "sheet order" in f["measured"]


def test_a_read_floor_order_keeps_the_fix_and_its_wording():
    s = with_stairs(base(), stair("A", "found", "missing", "found"))
    s["floors"]["not_checked"] = "No floor name was read on A2.9, so stairs were not followed there."
    f = only(s, 9)
    assert f["severity"] == "fix" and "sheet order" not in f["measured"]


def test_findings_reads_the_note_the_floors_stage_writes():
    with open(floors.__file__) as handle:
        assert findings.ORDER_NOT_READ in handle.read()


def test_gap_on_the_discharge_floor_raises_both_checks():
    s = with_stairs(base(), stair("A", "found", "missing", "found"))
    s["floors"]["discharge"] = 3
    assert only(s, 9)["severity"] == "fix" and only(s, 10)["severity"] == "fix"


def test_examiner_selector_moves_the_discharge_floor():
    s = with_stairs(base(), stair("A", "found", "end", "end"))
    assert raised(s, 10) == []
    s = with_stairs(base(), stair("A", "end", "found", "found"))
    assert raised(s, 10, discharge=3) == []
    assert only(s, 10)["sheet"] == 2


def test_unknown_discharge_makes_check_10_unknown():
    s = with_stairs(base(), stair("A", "end", "found", "found"))
    s["floors"]["discharge"] = None
    f = only(s, 10)
    assert f["severity"] == "unknown" and f["sheet"] is None


def test_selector_on_a_floor_not_followed_makes_check_10_unknown():
    assert only(with_stairs(base(), stair("A", "end", "found", "found")), 10, discharge=99)["severity"] == "unknown"


def test_no_stairs_followed_gives_no_discharge_item():
    s = with_stairs(base())
    s["floors"]["discharge"] = None
    assert raised(s, 10) == []


def test_every_stair_finding_names_its_crop():
    s = with_stairs(base(), stair("A", "end", "found", "found"), stair("B", "found", "missing", "found", pos=(4000.0, 800.0)), stair("C", "found", "found", "check", pos=(2000.0, 800.0)), stair("D", "found", "check", "found", pos=(3000.0, 800.0)))
    out = run(s)
    assert any(f["vertical"] for f in out["findings"])
    for f in [*out["findings"], *out["open_items"]]:
        if f["vertical"]:
            assert f"verticals/{f['vertical']}_" in f["evidence"]


def test_blank_cell_says_no_crop_was_cut():
    s = with_stairs(base(), stair("A", "found", "missing", "found"))
    s["verticals"]["verticals"][0]["floors"]["3"]["crop"] = None
    assert "No crop" in only(s, 9)["evidence"]


def unaligned_cell(s):
    c = s["verticals"]["verticals"][0]["floors"]["3"]
    c |= {"point": None, "crop": None}
    c["signals"] |= {"aligned": False, "spark": "0/3", "crop": None, "ink": None, "reason": "floor unaligned"}
    return s


def test_unaligned_floor_says_why_it_has_no_crop():
    s = unaligned_cell(with_stairs(base(), stair("A", "found", "check", "found")))
    evidence = only(s, 9)["evidence"]
    assert "No crop" in evidence and "not aligned" in evidence and "nothing is drawn" not in evidence


def test_a_signal_nobody_read_is_left_out_of_the_evidence():
    s = unaligned_cell(with_stairs(base(), stair("A", "found", "check", "found")))
    evidence = only(s, 9)["evidence"]
    assert "None" not in evidence and "Runs 0/3" in evidence and "crop answer" not in evidence and "ink" not in evidence


def test_evidence_gives_the_reason_for_the_state():
    assert "check here" in only(with_stairs(base(), stair("A", "found", "check", "found")), 9)["evidence"]


def test_a_cell_with_no_signals_reads_plainly():
    s = with_stairs(base(), stair("A", "found", "missing", "found"))
    s["verticals"]["verticals"][0]["floors"]["3"] |= {"crop": None, "signals": {}}
    assert only(s, 9)["evidence"] == "No crop, because nothing is drawn at this place."


def test_an_ink_reading_of_zero_stays_in_the_evidence():
    s = with_stairs(base(), stair("A", "found", "check", "found"))
    s["verticals"]["verticals"][0]["floors"]["3"]["signals"]["ink"] = 0.0
    assert "ink 0" in only(s, 9)["evidence"]


def test_stair_finding_has_a_location_without_a_mask():
    assert only(with_stairs(base(), stair("A", "found", "missing", "found")), 9)["box"]


# Check 11


def test_check_11_too_few_stairs_on_a_floor_is_check():
    s = with_stairs(base(), stair("A", "found", "found", "found"), stair("B", "found", "found", "end", pos=(4000.0, 800.0)))
    f = only(s, 11)
    assert f["severity"] == "check" and f["sheet"] == 4 and "1006.3.3" in f["rule"] and "2" in f["required"]


def test_check_11_ignores_the_discharge_floor_and_below():
    s = with_stairs(base(), stair("A", "found", "found", "found"), stair("B", "end", "found", "found", pos=(4000.0, 800.0)))
    assert raised(s, 11) == []


def test_check_11_a_check_cell_is_not_a_found_stair():
    s = with_stairs(base(), stair("A", "found", "found", "found"), stair("B", "found", "found", "check", pos=(4000.0, 800.0)))
    f = only(s, 11)
    assert f["sheet"] == 4 and "1 to confirm" in f["measured"]


def test_check_11_names_each_crop_it_counted():
    s = with_stairs(base(), stair("A", "found", "found", "found"), stair("B", "found", "found", "end", pos=(4000.0, 800.0)))
    assert "verticals/A_4.jpg" in only(s, 11)["evidence"]


def test_check_11_with_no_printed_load_is_unknown():
    s = base()
    s["declared"]["code"]["occupant_load"]["value"] = None
    assert only(s, 11)["severity"] == "unknown"


def test_check_11_with_unknown_discharge_is_unknown():
    s = base()
    s["floors"]["discharge"] = None
    assert only(s, 11)["severity"] == "unknown"


def test_check_11_roof_plan_is_not_a_storey():
    s = base()
    order = storeys(s, ["FIRST FLOOR", "SECOND FLOOR", "ROOF"])
    s["floors"]["discharge"] = order[0]
    with_stairs(s, stair("A", "found", "found", "end", order=order), stair("B", "found", "found", "end", pos=(4000.0, 800.0), order=order))
    s["declared"]["code"]["stories"]["value"] = "2"
    assert raised(s, 11) == []


def test_check_11_single_exit_group_needs_a_small_load():
    s = base()
    order = storeys(s, ["FIRST FLOOR", "SECOND FLOOR"])
    s["floors"]["discharge"] = order[0]
    with_stairs(s, stair("A", "found", "found", order=order))
    s["declared"]["code"]["stories"]["value"] = "2"
    s["declared"]["code"]["occupant_load"]["value"] = "29"
    assert raised(s, 11) == []
    s["declared"]["code"]["occupant_load"]["value"] = "30"
    assert only(s, 11)["severity"] == "check"


def test_check_11_a_floor_with_no_found_stair_gets_no_single_exit_case():
    s = base()
    order = storeys(s, ["FIRST FLOOR", "SECOND FLOOR"])
    s["floors"]["discharge"] = order[0]
    with_stairs(s, stair("A", "found", "end", order=order))
    s["declared"]["code"]["stories"]["value"] = "2"
    s["declared"]["code"]["occupant_load"]["value"] = "29"
    assert only(s, 11)["sheet"] == order[1]


def r2_stairs(s):
    order = storeys(s, ["FIRST FLOOR", "SECOND FLOOR", "THIRD FLOOR", "FOURTH FLOOR", "FIFTH FLOOR"])
    s["floors"]["discharge"] = order[0]
    with_stairs(s, stair("A", *["found"] * 5, order=order))
    s["declared"]["code"]["stories"]["value"] = "5"
    s["declared"]["code"]["occupancy"]["value"] = "R-2"
    s["declared"]["code"]["occupant_load"]["value"] = "40"
    return s, order


def test_seattle_r2_single_exit_differs_from_houston():
    s, order = r2_stairs(base())
    assert sorted(f["sheet"] for f in raised(s, 11) if f["severity"] == "check") == order[3:]
    seattle = raised(s, 11, jurisdiction="seattle")
    assert [f["severity"] for f in seattle] == ["unknown"]
    assert "item 7" in seattle[0]["rule"] and "Seattle" in seattle[0]["rule"] and "20 ft" in seattle[0]["required"]


def test_check_11_group_read_as_a_word_still_finds_its_rows():
    s, _ = r2_stairs(base())
    s["declared"]["code"]["occupancy"]["value"] = "Group R-2 / U"
    assert raised(s, 11, jurisdiction="seattle")[0]["severity"] == "check"   # U has no R-2 exception, so the stricter case stands


# Check 12


def test_top_and_bottom_of_each_stair_are_listed():
    items = opens(with_stairs(base(), stair("A", "end", "found", "found")))
    assert len(items) == 1
    assert "SECOND FLOOR" in items[0]["title"] and "THIRD FLOOR" in items[0]["title"] and items[0]["check"] == 12 and items[0]["vertical"] == "A"


def test_check_cell_at_the_end_of_a_run_is_an_open_item():
    s = with_stairs(base(), stair("A", "found", "found", "check"))
    assert raised(s, 9) == [] and raised(s, 10) == []
    assert any(item["sheet"] == 4 and "end of its run" in item["title"] for item in opens(s))


def test_stair_with_no_found_floor_still_reads_plainly():
    items = opens(with_stairs(base(), stair("F", "end", "end", "check")))
    assert sorted(i["title"] for i in items) == ["Stair F is found on no floor", "Stair F is not confirmed on THIRD FLOOR, at the end of its run"]
    assert all("verticals/F_4.jpg" in i["evidence"] for i in items)


def test_row_with_no_crop_says_so():
    s = with_stairs(base(), stair("G", "not_followed", "not_followed", "not_followed", established=False))
    for c in s["verticals"]["verticals"][0]["floors"].values():
        c["crop"] = None
    assert "No crop" in opens(s)[0]["evidence"]


def test_check_cell_on_discharge_is_not_also_an_open_item():
    s = with_stairs(base(), stair("A", "check", "found", "found"))
    assert not any(item["sheet"] == 2 and "not confirmed" in item["title"] for item in opens(s))


def test_not_followed_row_is_open_item():
    s = with_stairs(base(), stair("A", "found", "found", "found"), stair("G", "not_followed", "check", "not_followed", established=False, pos=(2000.0, 800.0)))
    item = next(i for i in opens(s) if i["vertical"] == "G")
    assert "one run" in item["title"]
    assert item["sheet"] == 3 and item["box"]
    assert run(s)["findings"] == [f for f in run(s)["findings"] if f["vertical"] != "G"]


def test_established_row_with_a_not_followed_cell_is_an_open_item():
    assert any("one run" in i["title"] for i in opens(with_stairs(base(), stair("A", "found", "not_followed", "found"))))


def test_row_seen_by_one_run_with_every_cell_to_confirm_is_an_open_item():
    assert any("one run" in i["title"] for i in opens(with_stairs(base(), stair("G", "check", "check", "check", established=False))))


def test_open_items_are_not_findings():
    assert all("severity" not in item for item in opens(base()))


# Check 13


def test_check_13_within_a_quarter_of_the_limit_is_check():
    s = base()
    s["vector"]["sheets"]["3"]["routes"] = [route(225.0)]
    f = only(s, 13)
    assert f["severity"] == "check" and f["sheet"] == 3 and "225" in f["measured"] and "300" in f["required"] and "Table 1017.2" in f["rule"]
    assert f["box"] == [100, 100, 100, 900]


def test_check_13_says_the_route_is_a_minimum():
    s = base()
    s["vector"]["sheets"]["3"]["routes"] = [route(225.0)]
    assert "at least" in only(s, 13)["measured"].lower()


def test_check_13_well_under_the_limit_raises_nothing():
    s = base()
    s["vector"]["sheets"]["3"]["routes"] = [route(224.9)]
    assert raised(s, 13) == []


def test_route_never_fix():
    s = base()
    s["vector"]["sheets"]["3"]["routes"] = [route(900.0)]
    assert only(s, 13)["severity"] == "check"


def test_check_13_not_permitted_case_is_check_at_any_length():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = "I-2"
    s["vector"]["sheets"]["3"]["routes"] = [route(10.0)]
    assert [f["severity"] for f in raised(s, 13, sprinklers="no")][:1] == ["check"]


def test_scan_distance_is_unknown():
    s = base()
    s["vector"]["sheets"]["2"] = blank(SCAN)
    f = only(s, 13)
    assert f["severity"] == "unknown" and f["sheet"] == 2 and SCAN in f["measured"]


def test_check_13_with_no_route_ends_is_unknown():
    s = base()
    s["vector"]["sheets"]["3"] |= {"routes": [], "skipped": NO_ENDS}
    assert only(s, 13)["severity"] == "unknown"


def test_check_13_trade_sheets_are_not_listed():
    s = base()
    s["vector"]["sheets"]["5"] = blank("Routes run only on the sheet chosen for each floor.")
    assert raised(s, 13) == []


def test_check_13_with_no_occupancy_read_is_unknown():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = None
    s["vector"]["sheets"]["3"]["routes"] = [route(225.0)]
    assert [f["severity"] for f in raised(s, 13)] == ["unknown"] * 3


def test_unknown_sprinklers_use_the_stricter_limit_and_ask():
    s = base()
    s["declared"]["code"]["sprinklers"]["value"] = None
    s["vector"]["sheets"]["3"]["routes"] = [route(150.0)]   # 75% of the unsprinklered 200 ft, 50% of 300 ft
    assert only(s, 13)["severity"] == "check"
    assert any("stricter" in f["title"] and f["severity"] == "unknown" for f in raised(s, 4))
    assert raised(s, 13, sprinklers="yes") == []


def test_no_route_compared_means_no_stricter_item():
    s = base()
    s["declared"]["code"]["sprinklers"]["value"] = None
    for i in ORDER:
        s["vector"]["sheets"][str(i)] = blank(SCAN)
    assert not any("stricter" in f["title"] for f in run(s)["findings"])


def test_code_sheet_statement_sets_sprinklers_with_no_selector():
    s = base()
    s["vector"]["sheets"]["3"]["routes"] = [route(150.0)]
    assert raised(s, 13) == []
    assert only(s, 13, sprinklers="no")["severity"] == "check"


def test_mixed_occupancy_with_a_not_permitted_group_is_check():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = "B, I-2"
    assert {f["severity"] for f in raised(s, 13, sprinklers="no")} == {"check"}


def test_mixed_occupancy_takes_the_strictest_limit():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = "B, S-2"
    s["vector"]["sheets"]["3"]["routes"] = [route(225.0)]
    assert "300" in only(s, 13)["required"]   # B at 300 ft, S-2 at 400 ft


# Check 14


def hall(clear, printed=None):
    return {"point": [1000, 1000], "clear_in": clear, "printed_in": printed}


def test_check_14_below_the_minimum_is_check():
    s = base()
    s["vector"]["sheets"]["3"]["corridors"] = [hall(41.0, 45.5)]
    f = only(s, 14)
    assert f["severity"] == "check" and f["sheet"] == 3 and "44" in f["required"] and "Table 1020.3" in f["rule"] and "45.5" in f["evidence"] and f["box"]


def test_check_14_printed_width_runs_wall_centre_to_wall_centre():
    s = base()
    s["vector"]["sheets"]["3"]["corridors"] = [hall(44.7, 49.0)]
    assert raised(s, 14) == []


def test_check_14_narrower_than_printed_by_more_than_a_wall_is_check():
    s = base()
    s["vector"]["sheets"]["3"]["corridors"] = [hall(50.0, 72.0)]
    f = only(s, 14)
    assert f["severity"] == "check" and "72" in f["required"]


def test_check_14_mixed_occupancy_takes_the_wider_minimum():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = "B, E"
    s["vector"]["sheets"]["3"]["corridors"] = [hall(60.0, 66.0)]
    assert "72" in only(s, 14)["required"]   # E at a load of 100 or more, B at 44 in


def test_check_14_no_reading_is_unknown():
    s = base()
    s["vector"]["sheets"]["3"]["corridors"] = [hall(None)]
    assert only(s, 14)["severity"] == "unknown"


def test_check_14_scan_width_is_unknown():
    s = base()
    s["vector"]["sheets"]["2"] = blank(SCAN)
    assert only(s, 14)["severity"] == "unknown"


def test_occupancy_read_as_a_word_reaches_its_rule_row():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = "BUSINESS"   # as real USC and clinic reads print it
    s["vector"]["sheets"]["3"]["corridors"] = [hall(41.0, 45.5)]
    assert "44" in only(s, 14)["required"]


def test_check_14_with_no_occupancy_read_is_unknown():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = None
    s["vector"]["sheets"]["3"]["corridors"] = [hall(41.0, 45.5)]
    assert [f["severity"] for f in raised(s, 14)] == ["unknown"] * 3


def test_check_14_floor_with_no_corridor_reading_is_unknown():
    s = base()
    s["vector"]["sheets"]["3"] |= {"corridors": [], "skipped": NO_ENDS}
    f = only(s, 14)
    assert f["severity"] == "unknown" and f["sheet"] == 3 and "SECOND FLOOR" in f["title"] and "orridor" in f["measured"] and NO_ENDS not in f["measured"]


# Check 16


def test_check_16_name_on_schedule_and_plan_differs():
    s = base()
    s["declared"]["rooms"] = [room("101", "WATER LAB")]
    f = only(s, 16)
    assert f["severity"] == "check" and f["sheet"] == 2 and "WATER LAB" in f["measured"] and "OFFICE" in f["measured"]


def test_check_16_typo_is_not_a_difference():
    s = base()
    s["declared"]["rooms"] = [room("101", "OFICE")]
    assert raised(s, 16) == []


def test_check_16_room_with_no_label_on_a_text_plan_is_check():
    s = base()
    s["declared"]["schedules"][1]["rows"].append(row("205", ROOM="BEDROOM 2"))
    f = only(s, 16)
    assert f["severity"] == "check" and f["sheet"] == 5 and "205" in f["title"]


def test_check_16_room_with_no_label_on_a_spark_plan_is_unknown():
    s = base()
    s["declared"]["schedules"][1]["rows"].append(row("205", ROOM="BEDROOM 2"))
    s["declared"]["rooms"] = [room("101", "Office", source="spark")]
    assert only(s, 16)["severity"] == "unknown"


def test_check_16_partial_room_labels_make_it_unknown():
    s = base()
    s["declared"]["schedules"][1]["rows"].append(row("205", ROOM="BEDROOM 2"))
    s["declared"]["cannot_tell"] = [{"what": "Room labels", "why": "Two reads did not agree on 205.", "sheet": 2}]
    assert {f["severity"] for f in raised(s, 16)} == {"unknown"}


def test_check_16_not_used_row_is_skipped():
    s = base()
    s["declared"]["schedules"][1]["rows"].append(row("205", ROOM="NOT USED"))
    assert raised(s, 16) == []


def test_check_16_plan_that_prints_names_only():
    s = base()
    s["declared"]["rooms"] = [room("", "OFFICE")]
    assert raised(s, 16) == []
    s["declared"]["schedules"][1]["rows"].append(row("205", NAME="BED"))
    assert only(s, 16)["severity"] == "check"


def test_check_16_cell_two_reads_differ_on_is_skipped():
    s = base()
    s["declared"]["schedules"][1]["rows"].append(row("101", NAME=None))
    assert raised(s, 16) == []


def test_check_16_door_schedule_room_against_plan_label():
    s = base()
    s["declared"]["schedules"].append(schedule(5, "door", [row("101A", WIDTH="36", ROOM="WATER LAB"), row("101B", WIDTH="36", ROOM="WATER LAB")], [], source="spark"))
    f = only(s, 16)
    assert "WATER LAB" in f["measured"] and "Office" in f["measured"]


def test_check_16_modules_that_reuse_room_numbers_do_not_cross():
    s = base()
    s["read"]["sheets"] += [sheet(6, "B2.0", "MODULE B FLOOR PLANS"), sheet(7, "B6.0", "MODULE B SCHEDULES", "schedule")]
    s["declared"]["schedules"].append(schedule(7, "finish", [row("101", ROOM="BEDROOM1")], []))
    s["declared"]["rooms"].append(room("101", "BEDROOM1", sheet_index=6))
    assert raised(s, 16) == []


def test_check_16_schedule_on_a_plan_sheet_reads_that_sheet_first():
    s = base()
    s["read"]["sheets"] += [sheet(6, "A2.4", "FOURTH FLOOR PLAN", level="FOURTH FLOOR")]
    s["declared"]["schedules"].append(schedule(6, "finish", [row("101", ROOM="STORE")], []))
    s["declared"]["rooms"].append(room("101", "STORE", sheet_index=6))
    assert raised(s, 16) == []


def test_check_16_schedule_with_no_plan_of_its_letter_still_compares():
    s = base()
    s["read"]["sheets"][5]["number"] = "F6.1"
    s["declared"]["rooms"] = [room("101", "WATER LAB")]
    assert only(s, 16)["sheet"] == 2


def test_check_16_two_modules_can_each_have_a_room_that_differs():
    s = base()
    s["read"]["sheets"] += [sheet(6, "B2.0", "MODULE B FLOOR PLANS"), sheet(7, "B6.0", "MODULE B SCHEDULES", "schedule")]
    s["declared"]["schedules"].append(schedule(7, "door", [row("101A", ROOM="WATER LAB")], [], source="spark"))
    s["declared"]["schedules"].append(schedule(7, "finish", [row("101", ROOM="BEDROOM1")], []))
    s["declared"]["rooms"] = [room("101", "WATER LAB"), room("101", "OFFICE", sheet_index=6)]
    assert sorted(f["sheet"] for f in raised(s, 16)) == [2, 6]


def test_no_room_labels_compares_nothing():
    s = base()
    s["declared"]["rooms"] = []
    assert raised(s, 16) == []


def test_every_dropped_check_stays_dropped():
    s = base()
    s["declared"]["exits"]["sheets"] = [{"sheet": 1, "sum": None, "reads": [0, 0], "markers": [{"point": [10, 10], "width_mm": 915.0, "capacity": None}]}]
    s["declared"]["rooms"].append(room("999", "CLOSET"))   # plan room not in schedule
    s["declared"]["schedules"][0]["rows"][1]["cells"] |= {"SWING": "PAIR"}   # closed swing vocabulary
    s["declared"]["schedules"].append(schedule(5, "door", [row("777", WIDTH="36", ROOM="CLOSET")], [], source="spark"))   # door room not in finish
    s["declared"]["code"]["occupancy"]["mentions"] = ["INSTITUTIONAL OUTPATIENT"]   # occupancy word match
    s["declared"]["titles"][2] = title(2, "A2.1", "FLOOR 1", judge="same")   # title compare by normaliser alone
    s["vector"]["sheets"]["2"]["doors"] = [door(width_in=30.0)]   # width against arc, row says 36
    assert run(s)["findings"] == []


def test_swing_with_no_arc_raises_nothing():
    s = base()
    s["vector"]["sheets"]["2"]["doors"] = []
    assert raised(s, 7) == []


# Reads that did not finish


def test_unknown_read_gives_unknown_not_silence():
    s = base()
    notes = [("Text layer", None, 2), ("Sheet number", 3, 2), ("Sheet title", 3, 3), ("Label under the drawing", 3, 3), ("Code summary", None, 4), ("Code field stories", 0, 4), ("Door schedule", 5, 7), ("Window schedule", 5, 7),
             ("Door tags", 2, 7), ("Plan not read", 2, 7), ("Finish schedule", 5, 16), ("Room labels", 2, 16), ("Occupant table", 0, 6), ("Exit table", 0, 6), ("Area table", 0, 6), ("Exit capacity", 1, 8), ("Sheet index", 0, 2),
             ("Sheets not placed", None, 1), ("Spark reads", None, 1)]
    s["declared"]["cannot_tell"] = [{"what": what, "why": f"reason {n}.", "sheet": sheet_index} for n, (what, sheet_index, _) in enumerate(notes)]
    out = run(s)
    for n, (what, sheet_index, check) in enumerate(notes):
        hits = [f for f in out["findings"] if f"reason {n}." in f["title"]]
        assert len(hits) == 1 and hits[0]["severity"] == "unknown" and hits[0]["sheet"] == sheet_index and hits[0]["check"] == check and what in hits[0]["title"], what


def test_two_notes_on_one_sheet_keep_their_own_keys():
    s = base()
    s["declared"]["cannot_tell"] = [{"what": "Sheet title", "why": "first", "sheet": 2}, {"what": "Sheet title", "why": "second", "sheet": 2}]
    keys = [f["key"] for f in raised(s, 3)]
    assert len(set(keys)) == 2


@pytest.mark.parametrize("stage", ["read", "floors", "verticals", "declared", "vector"])
def test_failed_stage_appears_in_not_checked(stage):
    s = base()
    s[stage] = {"failed": "The review reached its spend cap."}
    out = run(s)
    line = next(line for line in out["not_checked"] if "The review reached its spend cap." in line)
    assert stage in line
    assert [f for f in out["findings"] if f["severity"] == "fix"] == []   # nothing read is not the same as nothing there


@pytest.mark.parametrize("stage", ["read", "floors", "verticals", "declared", "vector"])
def test_stage_with_no_result_counts_as_not_finished(stage):
    s = base()
    s[stage] = {}
    out = run(s)
    assert any(stage in line for line in out["not_checked"])
    assert [f for f in out["findings"] if f["severity"] == "fix"] == []


def test_missing_stage_key_counts_as_not_finished():
    s = base()
    del s["vector"]
    assert any("vector" in line for line in run(s)["not_checked"])


@pytest.mark.parametrize("order", [[2], []], ids=["one floor plan", "no floor plans"])
def test_set_with_too_few_floors_finishes_with_its_note(order):
    s = base()
    note = "Stairs were not followed across floors, because this set has one floor plan."
    s["floors"] |= {"order": order, "not_checked": note, "stack": None, "discharge": None}
    s["verticals"] = {"verticals": [], "not_checked": [SHAFTS, note]}
    assert note in run(s)["not_checked"]


LOST_CHECKS = ("travel distance", "corridor width")


def no_order(stages, note):
    stages["floors"] |= {"order": [], "reference": None, "stack": None, "frame_size": None, "discharge": None, "not_checked": note}
    stages["verticals"] = {"verticals": [], "not_checked": [SHAFTS, note]}
    for i in ORDER:
        stages["vector"]["sheets"][str(i)] |= {"routes": [], "skipped": "No floor order was found, so routes did not run."}
    return stages


def left_out(stages):
    return [f for f in raised(stages, 1) if "not checked on" in f["title"]]


def test_unread_middle_floor_leaves_a_visible_item():
    s = base()
    unread(s, 3)
    s["floors"]["order"] = [2, 4]
    with_stairs(s, stair("A", "found", "found", order=[2, 4]), stair("B", "found", "found", pos=(4000.0, 800.0), order=[2, 4]))
    f = next(f for f in run(s)["findings"] if f["sheet"] == 3 and f["check"] == 1)
    assert f["severity"] == "unknown"
    assert all(word in f["measured"] for word in ("stairs", *LOST_CHECKS))


def test_each_unread_sheet_is_its_own_item():
    s = base()
    unread(s, 3)
    unread(s, 5)
    keys = [f["key"] for f in raised(s, 1) if "could not be read" in f["title"]]
    assert len(set(keys)) == 2


def test_unread_sheet_stays_visible_when_the_declared_stage_failed():
    s = base()
    unread(s, 3)
    s["declared"] = {"failed": "Stopped."}
    assert [f["sheet"] for f in raised(s, 1)] == [3]


@pytest.mark.parametrize("level, note", [("", "No floor name was read on A2.1, so stairs were not followed there."), ("FIRST FLOOR", "Stairs were not followed across floors, because this set has 51 floor plans and the limit is 50.")], ids=["no floor name", "over the floor limit"])
def test_empty_order_names_each_plan_and_what_it_lost(level, note):
    s = base()
    for r in s["read"]["sheets"][2:5]:
        r["level"] = level
    no_order(s, note)
    s["vector"]["sheets"]["2"]["corridors"] = [hall(30.0, 34.5)]
    f = only(s, 1)
    assert f["severity"] == "unknown" and f["sheet"] is None
    assert all(number in f["title"] and number in f["measured"] for number in ("A2.1", "A2.2", "A2.3"))
    assert all(word in f["title"].lower() for word in ("stairs", *LOST_CHECKS))


def test_plan_dropped_as_a_repeat_is_named_with_what_it_lost():
    s = base()
    s["read"]["sheets"][3]["level"] = "FIRST FLOOR"
    s["floors"]["order"] = [2, 4]
    s["floors"]["not_checked"] = "A2.2 read the same floor name as another sheet, so stairs were not followed there."
    with_stairs(s, stair("A", "found", "found", order=[2, 4]), stair("B", "found", "found", pos=(4000.0, 800.0), order=[2, 4]))
    s["vector"]["sheets"]["3"]["corridors"] = [hall(30.0, 34.5)]
    f = left_out(s)[0]
    assert "A2.2" in f["title"] and not {"A2.1", "A2.3"} & set(f["title"].replace(",", "").split())
    assert s["floors"]["not_checked"] in run(s)["not_checked"]


def test_long_list_of_plans_keeps_every_name_in_measured():
    s = base()
    levels = [f"LEVEL {n}" for n in range(1, 14)]
    order = storeys(s, levels)
    no_order(s, "Stairs were not followed across floors, because this set has 51 floor plans and the limit is 50.")
    f = only(s, 1)
    assert "and 3 more" in f["title"] and all(f"A3.{i}" in f["measured"] for i in order)


@pytest.mark.parametrize("row", [sheet(6, "A1.2", "LIFE SAFETY PLAN", level="SECOND FLOOR"), sheet(6, "A2.4", "ROOF PLAN"), sheet(6, "E2.1", "POWER PLAN"), sheet(6, "A2.9", "NOTES", "other")], ids=["life safety copy", "roof", "trade sheet", "not a plan"])
def test_sheets_that_lost_no_check_are_not_listed(row):
    s = base()
    s["read"]["sheets"].append(row)
    assert left_out(s) == []


def test_unnamed_plan_with_a_copy_title_is_listed_as_the_floors_note_lists_it():
    s = base()
    s["read"]["sheets"].append(sheet(6, "A2.4", "PARTIAL FLOOR PLAN"))
    assert "A2.4" in left_out(s)[0]["title"]


def test_plans_are_not_listed_when_the_floors_stage_failed():
    s = base()
    s["floors"] = {"failed": "Stopped."}
    assert left_out(s) == []


def test_every_plan_in_the_order_lists_nothing():
    assert left_out(base()) == []


def test_failed_stage_leaves_other_checks_running():
    s = base()
    s["vector"] = {"failed": "Measuring did not finish."}
    s["declared"]["tags"].append(tag(2, "13"))
    assert only(s, 7)["severity"] == "fix"


def test_check_that_crashes_loses_only_itself():
    s = base()
    s["declared"]["tags"].append(tag(2, "13"))
    s["declared"]["tables"] = [{"sheet": 0, "kind": "occupant", "rows": None, "box": None}]
    out = evaluate(s, "houston", None, None)
    assert [f["check"] for f in out["findings"]] == [7]
    assert any("table arithmetic" in line for line in out["not_checked"])


def test_unknown_jurisdiction_is_an_error_not_an_empty_review():
    with pytest.raises(KeyError):
        run(base(), jurisdiction="gotham")


def test_fixed_not_checked_lines_are_always_there():
    for s in (base(), base() | {"verticals": {"failed": "Stopped."}}):
        lines = run(s)["not_checked"]
        assert SHAFTS in lines
        assert any("Electrical" in line and "not reviewed" in line for line in lines)
        assert lines.count(SHAFTS) == 1


def test_floors_note_and_verticals_notes_both_reach_not_checked():
    s = base()
    s["floors"]["not_checked"] = "No floor name was read on A2.9, so stairs were not followed there."
    s["verticals"]["not_checked"] = [SHAFTS, "1 of the 3 stair comparison runs gave no usable reply, so the stair matrix uses the others."]
    lines = run(s)["not_checked"]
    assert s["floors"]["not_checked"] in lines and s["verticals"]["not_checked"][1] in lines


def test_a_floors_note_copied_by_verticals_is_listed_once():
    s = base()
    note = "Stairs were not followed across floors, because this set has one floor plan."
    s["floors"]["not_checked"] = note
    s["verticals"] = {"verticals": [], "not_checked": [SHAFTS, note]}
    assert run(s)["not_checked"].count(note) == 1


def test_no_pass_value_anywhere():
    s = with_stairs(base(), stair("A", "found", "missing", "found", name="APPROVED STAIR"), stair("B", "end", "found", "found", pos=(4000.0, 800.0)), stair("C", "not_followed", "check", "not_followed", established=False, pos=(2000.0, 800.0), name="passes"))
    planted = "THIS SET IS APPROVED, COMPLIANT, PASSED. PASS. PASSES. IT COMPLIES WITH CODE COMPLIANCE, APPROVAL GRANTED"
    d = s["declared"]
    s["read"]["sheets"][3]["level"] = "APPROVED LEVEL"
    d["rooms"] = [room("101", planted)]
    d["titles"][2] = title(2, "A2.1", f"MAIN FLOOR PLAN {planted}", under=f"Ground Floor Plan {planted}")
    d["index"][5]["title"] = planted
    d["titles"][5] = title(5, "A6.1", f"PLUMBING {planted}", judge="different")
    d["index"].append({"number": "A9.9", "title": planted, "not_used": False, "sheet": 0, "box": None})
    d["tags"] += [tag(2, "PASS"), tag(2, "APPROVED")]
    d["schedules"][0]["rows"].append(row("OK", SWING="BIFOLD", SIZE=planted))
    d["schedules"][1]["rows"] += [row("205", ROOM=planted), row("206", ROOM="OFFICE")]
    d["code"]["edition"]["value"] = f"2018 {planted}"
    d["code"]["occupancy"]["value"] = planted
    d["tables"] = [occupant_table(load_row(4351.0, 100.0, 43) | {"name": planted, "room": "PASS"})]
    d["exits"]["sheets"] = [exit_sheet(100)]
    d["cannot_tell"] = [{"what": planted, "why": planted, "sheet": 2}]
    s["vector"]["sheets"]["3"]["routes"] = [route(290.0) | {"from_room": planted, "to_exit": planted}]
    s["vector"]["sheets"]["4"] = blank(planted)
    s["vector"]["sheets"]["2"]["doors"] = [door()]
    s["floors"]["not_checked"] = planted
    s["verticals"]["not_checked"] = [planted]
    out = run(s) | {"also": run(s | {"read": {"failed": planted}})}
    assert out["findings"] and out["open_items"] and out["not_checked"]
    assert not [text for text in strings(out) if STATUS.search(text)]
    assert not STATUS.search(json.dumps(out))


def test_room_named_passage_is_left_alone():
    s = base()
    s["declared"]["rooms"] = [room("101", "PASSAGE")]
    assert "PASSAGE" in only(s, 16)["measured"]


def test_no_finding_state_is_a_pass():
    s = with_stairs(base(), stair("A", "found", "missing", "found"))
    assert {f["severity"] for f in run(s)["findings"]} <= {"fix", "check", "unknown"}


def test_evaluate_makes_no_model_call(monkeypatch):
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("evaluate reached the real client"))
    meta.set_transport(lambda kind, **request: pytest.fail("evaluate called the model"))
    try:
        assert run(base())["findings"] == []
    finally:
        meta.set_transport(None)


def test_evaluate_leaves_its_input_alone():
    s = with_stairs(base(), stair("A", "found", "missing", "found"))
    before = json.dumps(s, sort_keys=True)
    run(s)
    assert json.dumps(s, sort_keys=True) == before


def test_output_is_plain_json():
    s = with_stairs(base(), stair("A", "found", "missing", "found"))
    out = run(s)
    assert json.loads(json.dumps(out, allow_nan=False)) == out


def test_findings_lead_with_the_strongest():
    s = base()
    s["declared"]["cannot_tell"] = [{"what": "Sheet index", "why": "first", "sheet": 0}]   # unknown, from the first stage it reads
    s["declared"]["titles"][2] = title(2, "A2.1", "MAIN FLOOR PLAN", under="Ground Floor Plan")   # check, in check 3
    s["declared"]["tags"].append(tag(2, "13"))   # fix, in check 7
    assert [f["severity"] for f in run(s)["findings"]] == ["fix", "check", "unknown"]


# Dwellings


DWELLING_CHECKS = (11, 13, 14)


def dwelling():
    s = base()
    s["declared"]["code"]["edition"]["value"] = "2021 International Residential Code"
    s["declared"]["code"]["occupancy"]["value"] = None
    s["declared"]["code"]["occupant_load"]["value"] = None
    s["declared"]["exits"]["sheets"] = []
    s["read"]["sheets"][1] |= {"title": "FIRST FLOOR LAYOUT", "cls": "detail"}
    s["declared"]["titles"][1]["title"] = "FIRST FLOOR LAYOUT"
    s["declared"]["index"][1]["title"] = "FIRST FLOOR LAYOUT"
    s["vector"]["sheets"]["3"]["routes"] = [route(900.0)]
    s["vector"]["sheets"]["3"]["corridors"] = [hall(30.0, 34.5)]
    return s


def test_dwelling_moves_rule_checks_to_not_checked():
    out = run(dwelling())
    assert [f for f in out["findings"] if f["check"] in DWELLING_CHECKS] == []
    assert raised(dwelling(), 1) == []   # no life safety sheet is asked for
    line = next(line for line in out["not_checked"] if "dwelling" in line)
    for name in ("life safety", "exits", "travel distance", "corridor width"):
        assert name in line


def test_dwelling_keeps_coordination_checks():
    s = dwelling()
    s["declared"]["tags"].append(tag(2, "13"))   # 7
    s["declared"]["index"].pop()   # 2
    s["declared"]["titles"][3] = title(3, "A2.2", "MAIN FLOOR PLAN", under="Ground Floor Plan")   # 3
    s["declared"]["tables"] = [occupant_table(load_row(4351.0, 100.0, 43))]   # 6
    with_stairs(s, stair("A", "end", "missing", "found"), stair("B", "found", "missing", "found", pos=(4000.0, 800.0)))   # 10, 9
    assert {f["check"] for f in run(s)["findings"]} >= {2, 3, 6, 7, 9, 10}


def test_dwelling_edition_is_not_compared_with_the_building_code():
    s = dwelling()
    s["declared"]["code"]["edition"]["value"] = "2018 International Residential Code"
    out = run(s, jurisdiction="seattle")
    assert [f for f in out["findings"] if f["check"] == 5] == []
    assert any("code edition" in line for line in out["not_checked"] if "dwelling" in line)


def test_dwelling_still_compares_stories_with_its_plans():
    s = dwelling()
    s["declared"]["code"]["stories"]["value"] = "5 STORIES"
    assert only(s, 5)["severity"] == "fix"


def test_dwelling_is_not_asked_for_fields_the_residential_code_does_not_have():
    s = dwelling()
    s["declared"]["code"]["construction"]["value"] = None
    assert [f for f in raised(s, 4) if any(w in f["title"] for w in ("occupancy", "construction", "occupant load"))] == []
    assert any("occupant load" in line for line in run(s)["not_checked"] if "dwelling" in line)


def test_dwelling_is_still_asked_for_sprinklers():
    s = dwelling()
    s["declared"]["code"]["sprinklers"]["value"] = None
    assert [f["severity"] for f in raised(s, 4) if "sprinkler" in f["title"]] == ["check"]


def test_dwelling_shown_by_room_labels():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = None
    s["declared"]["code"]["edition"]["value"] = "2021 CBC"
    s["declared"]["rooms"] = [room("101", "Office"), room("102", "Bedroom 2")]
    assert any("dwelling" in line for line in run(s)["not_checked"])


@pytest.mark.parametrize("field, value", [("occupancy", "R-3"), ("occupancy", "SINGLE FAMILY RESIDENCE"), ("occupancy", "R-3 / U"), ("edition", "2021 IRC"), ("edition", "2022 California Residential Code")])
def test_dwelling_shown_by_the_code_sheet(field, value):
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = None
    s["declared"]["code"][field]["value"] = value
    assert any("dwelling" in line for line in run(s)["not_checked"])


@pytest.mark.parametrize("occupancy, edition", [("SINGLE FAMILY RESIDENTIAL", "2021 IRC"), ("RESIDENTIAL", "2021 International Residential Code"), ("GROUP R", "2021 IRC")])
def test_a_bare_r_does_not_outrank_a_residential_statement(occupancy, edition):
    s = dwelling()
    s["declared"]["code"]["occupancy"]["value"] = occupancy
    s["declared"]["code"]["edition"]["value"] = edition
    out = run(s)
    assert any("dwelling" in line for line in out["not_checked"])
    assert [f for f in out["findings"] if f["check"] in (1, *DWELLING_CHECKS)] == []


@pytest.mark.parametrize("occupancy", ["M / R-3", "B, R-3", "BUSINESS"])
def test_a_group_with_no_digit_still_outranks_a_residential_code_name(occupancy):
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = occupancy
    s["declared"]["code"]["edition"]["value"] = "2021 IRC"
    assert not any("dwelling" in line for line in run(s)["not_checked"])


def test_group_r_alone_is_no_dwelling():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = "GROUP R"
    s["declared"]["rooms"] = [room("101", "Office"), room("102", "Bedroom 2")]   # an apartment block has bedrooms too
    assert not any("dwelling" in line for line in run(s)["not_checked"])


def test_a_lost_occupancy_read_is_not_a_dwelling():
    s = base()
    s["declared"]["code"]["occupancy"] |= {"value": None, "reads": 0}
    s["declared"]["cannot_tell"] = [{"what": "Code field occupancy", "why": "A read of this sheet was lost.", "sheet": 0}]
    s["declared"]["rooms"] = [room("101", "Office"), room("102", "Bedroom 2")]
    s["vector"]["sheets"]["3"]["routes"] = [route(900.0)]
    out = run(s)
    assert not any("dwelling" in line for line in out["not_checked"])
    assert [f for f in out["findings"] if f["check"] == 13]


def test_one_read_of_the_occupancy_is_not_enough_for_room_labels():
    s = base()
    s["declared"]["code"]["occupancy"] |= {"value": None, "reads": 1}
    s["declared"]["rooms"] = [room("101", "Office"), room("102", "Bedroom 2")]
    assert not any("dwelling" in line for line in run(s)["not_checked"])


def test_dwelling_shown_by_room_labels_with_no_code_sheet():
    s = base()
    s["declared"]["code"] = {}
    s["declared"]["cannot_tell"] = [{"what": "Code summary", "why": "No code sheet was found in this set.", "sheet": None}]
    s["declared"]["rooms"] = [room("101", "Office"), room("102", "Bedroom 2")]
    assert any("dwelling" in line for line in run(s)["not_checked"])


def test_declared_ibc_group_outranks_a_residential_code_name():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = "R-2"
    s["declared"]["code"]["edition"]["value"] = "2021 IRC"
    assert not any("dwelling" in line for line in run(s)["not_checked"])


def test_declared_apartment_block_is_not_a_dwelling():
    s = base()
    s["declared"]["code"]["occupancy"]["value"] = "R-2"
    s["declared"]["rooms"] = [room("101", "Office"), room("102", "Bedroom 2")]
    assert not any("dwelling" in line for line in run(s)["not_checked"])


# Keys and comment ids


def gap(stair_id, pos, name, states=("found", "missing", "found")):
    return with_stairs(base(), stair(stair_id, *states, pos=pos, name=name))


def test_key_is_stable_across_runs():
    first, second = [only(gap(*run_row), 9) for run_row in (("F", (5079.8, 2161.6), "Stair_W_North"), ("E", (4986.5, 2049.6), "central_west_main"))]   # one real stair on two fresh bme_k runs, 146 px apart
    assert first["key"] == second["key"]


def test_corridor_key_holds_its_rounded_cell():
    def key(point):
        s = base()
        s["vector"]["sheets"]["3"]["corridors"] = [hall(41.0) | {"point": list(point)}]
        return only(s, 14)["key"]

    assert key((1000, 1000)) != key((4000, 1000))
    assert key((5079.8, 2161.6)) == key((4986.5, 2049.6))


def test_key_is_the_same_for_every_evaluate_of_one_stage_file():
    s = base()
    s["declared"]["tags"].append(tag(2, "13"))
    assert only(s, 7)["key"] == only(s, 7)["key"]


def test_key_does_not_change_with_jurisdiction():
    s = gap("A", (1000.0, 800.0), "Stair")
    assert only(s, 9)["key"] == only(s, 9, jurisdiction="seattle")["key"]


def test_key_differs_by_check_sheet_and_place():
    s = with_stairs(base(), stair("A", "found", "missing", "found"), stair("B", "found", "missing", "found", pos=(4000.0, 800.0)))
    s["floors"]["discharge"] = 3
    keys = [f["key"] for f in run(s)["findings"]]
    assert len(keys) == len(set(keys)) == 4


def test_two_stairs_in_one_grid_cell_keep_separate_keys():
    s = with_stairs(base(), stair("A", "found", "missing", "found"), stair("B", "found", "missing", "found", pos=(1040.0, 810.0)))
    keys = [f["key"] for f in raised(s, 9)]
    assert len(set(keys)) == 2


def test_two_stairs_in_one_grid_cell_keep_separate_open_item_keys():
    s = with_stairs(base(), stair("A", "found", "found", "found"), stair("B", "found", "found", "found", pos=(1040.0, 810.0)))
    keys = [item["key"] for item in opens(s)]
    assert len(keys) == len(set(keys)) == 2


def test_assign_ids_numbers_in_order():
    findings_ = [{"key": "a"}, {"key": "b"}]
    assign_ids(findings_, {})
    assert [f["comment_id"] for f in findings_] == ["A-01", "A-02"]


def test_assign_ids_keeps_a_known_id_and_skips_used_numbers():
    findings_ = [{"key": "new"}, {"key": "known"}]
    assign_ids(findings_, {"known": "A-07", "resolved": "A-09"})
    assert [f["comment_id"] for f in findings_] == ["A-10", "A-07"]


def test_a_stair_that_drifted_across_a_grid_line_keeps_its_id_and_key():
    first = only(gap("B", (3400.0, 760.0), "Stair_W_North"), 9)
    second = only(gap("G", (3546.0, 760.0), "central_west_main"), 9)   # 146 px, the largest drift between two fresh bme_k runs
    assert first["key"] != second["key"]
    assign_ids([first], {})
    assign_ids([second], {first["key"]: first["comment_id"]})
    assert second["comment_id"] == first["comment_id"] and second["key"] == first["key"]


def test_two_drifted_stairs_each_take_the_nearest_old_comment():
    old = [only(gap("A", (1200.0, 600.0), "a"), 9), only(gap("B", (1500.0, 1200.0), "b"), 9)]
    assign_ids(old, {})
    previous = {f["key"]: f["comment_id"] for f in old}
    new = [only(gap("C", (1500.0, 900.0), "b"), 9), only(gap("D", (1200.0, 900.0), "a"), 9)]   # each one cell from both old comments, and nearer its own
    assign_ids(new, previous)
    assert [f["comment_id"] for f in new] == [old[1]["comment_id"], old[0]["comment_id"]]


@pytest.mark.parametrize("order", ["EI", "IE"])
def test_ids_do_not_follow_finding_order_on_real_stairs(order):
    old = [only(gap("F", (5079.8, 2161.6), "a"), 9), only(gap("G", (5115.3, 2759.4), "b"), 9)]   # one bme_k run
    assign_ids(old, {})
    new = {"E": only(gap("E", (4986.5, 2049.6), "c"), 9), "I": only(gap("I", (4999.3, 2676.0), "d"), 9)}   # the same two stairs on the next, where row ids can come in either order
    assign_ids([new[row] for row in order], {f["key"]: f["comment_id"] for f in old})
    assert (new["E"]["comment_id"], new["I"]["comment_id"]) == ("A-01", "A-02")


@pytest.mark.parametrize("flip", [False, True], ids=["nearest first", "nearest last"])
def test_the_nearest_stair_takes_an_old_comment_in_any_finding_order(flip):
    old = only(gap("A", (1200.0, 900.0), "a"), 9)
    assign_ids([old], {})
    near, far = only(gap("B", (1500.0, 900.0), "b"), 9), only(gap("C", (900.0, 600.0), "c"), 9)   # one cell along, and one cell diagonally with the key that sorts first
    assign_ids([far, near] if flip else [near, far], {old["key"]: old["comment_id"]})
    assert (near["comment_id"], far["comment_id"]) == (old["comment_id"], "A-02")


def test_one_stair_near_two_old_comments_takes_the_nearer():
    old = [only(gap("A", (1200.0, 900.0), "a"), 9), only(gap("B", (1500.0, 900.0), "b"), 9)]
    assign_ids(old, {})
    new = only(gap("C", (1200.0, 1200.0), "c"), 9)   # one cell along from the first, one cell diagonally from the second
    assign_ids([new], {f["key"]: f["comment_id"] for f in old})
    assert new["comment_id"] == old[0]["comment_id"]


def test_stairs_equally_near_an_old_comment_get_one_answer_in_any_finding_order():
    def ids(flip):
        old = only(gap("A", (1200.0, 900.0), "a"), 9)
        assign_ids([old], {})
        left, right = only(gap("B", (900.0, 900.0), "b"), 9), only(gap("C", (1500.0, 900.0), "c"), 9)
        assign_ids([right, left] if flip else [left, right], {old["key"]: old["comment_id"]})
        return left["comment_id"], right["comment_id"]

    assert ids(False) == ids(True)


def test_an_old_comment_goes_to_one_stair_only():
    old = only(gap("A", (1000.0, 800.0), "a"), 9)
    assign_ids([old], {})
    new = [only(gap("B", (1000.0, 800.0), "b"), 9), only(gap("C", (1210.0, 800.0), "c"), 9)]
    assign_ids(new, {old["key"]: old["comment_id"]})
    assert new[0]["comment_id"] == old["comment_id"] and new[1]["comment_id"] == "A-02" and new[1]["key"] != old["key"]


def test_two_stairs_cannot_both_drift_onto_one_old_comment():
    old = only(gap("A", (1000.0, 800.0), "a"), 9)
    assign_ids([old], {})
    new = [only(gap("B", (1210.0, 800.0), "b"), 9), only(gap("C", (740.0, 800.0), "c"), 9)]
    assign_ids(new, {old["key"]: old["comment_id"]})
    assert sorted(f["comment_id"] for f in new) == [old["comment_id"], "A-02"]


def test_a_stair_two_cells_from_every_old_comment_gets_a_new_id():
    old = only(gap("A", (1000.0, 800.0), "a"), 9)
    assign_ids([old], {})
    new = only(gap("B", (1500.0, 800.0), "b"), 9)
    assign_ids([new], {old["key"]: old["comment_id"]})
    assert new["comment_id"] == "A-02"


def test_drift_match_stays_within_one_check_and_sheet():
    s = gap("A", (1000.0, 800.0), "a", states=("end", "found", "found"))
    old = only(s, 10)
    assign_ids([old], {})
    new = only(gap("B", (1040.0, 800.0), "b"), 9)
    assign_ids([new], {old["key"]: old["comment_id"]})
    assert new["comment_id"] == "A-02"


def test_module_has_no_file_or_network_use():
    source = findings.__file__
    with open(source) as handle:
        text = handle.read()
    assert not re.search(r"\bopen\(|read_text|read_json|spark\(|sam_video\(|sam_image\(", text)
