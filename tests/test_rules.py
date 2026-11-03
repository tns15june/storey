import json
import re
from pathlib import Path

import pytest

from storey.rules import JURISDICTIONS, ROWS, SYSTEMS, rule, stricter_when_unknown

RESEARCH = Path(__file__).resolve().parent.parent / "data/rules/ibc2021_egress.json"
T1, T2 = "ibc.1006.3.4.t1", "ibc.1006.3.4.t2"
REDUCED = {"occupancy": "B", "sprinklers": "NFPA 13", "voice_alarm": "yes"}
GROUPS = "A-1 A-2 A-3 A-4 A-5 B E F-1 F-2 H-1 H-2 H-3 H-4 H-5 I-1 I-2 I-3 I-4 M R-1 R-2 R-3 R-4 S-1 S-2 U".split()
PROBES = [  # rule id, research table row, research field, inputs that should land on that cell
    ("ibc.1005.3.1", 0, "factor", {}),
    ("ibc.1005.3.1", 1, "factor", REDUCED),
    ("ibc.1005.3.2", 0, "factor", {}),
    ("ibc.1005.3.2", 1, "factor", REDUCED),
    ("ibc.1006.3.3", 0, "min_exits", {"occupant_load": 500}),
    ("ibc.1006.3.3", 1, "min_exits", {"occupant_load": 1000}),
    ("ibc.1006.3.3", 2, "min_exits", {"occupant_load": 1001}),
    (T1, 1, "max_dwelling_units", {"story": 4, "sprinklers": "NFPA 13"}),
    ("ibc.1010.1.1", 0, "min", {"occupancy": "B"}),
    ("ibc.1010.1.1", 1, "min", {"occupancy": "I-2"}),
    ("ibc.1020.3", 0, "min_width", {"occupancy": "B", "occupant_load": 50}),
    ("ibc.1020.3", 2, "min_width", {"occupancy": "B", "occupant_load": 49}),
    ("ibc.1020.3", 4, "min_width", {"occupancy": "E", "occupant_load": 100}),
    ("ibc.1020.3", 5, "min_width", {"occupancy": "B", "occupant_load": 50, "use": "stretcher traffic"}),
    ("ibc.1020.3", 6, "min_width", {"occupancy": "I-2", "occupant_load": 50}),
    ("ibc.1020.5", 0, "max_length", {}),
]


def all_cases(row):
    return row["cases"] + [case for cases in row.get("overrides", {}).values() for case in cases]


def listed(label, group):
    """The research table's occupancy cell names this group or its division."""
    tokens = label.split(", ")
    return group in tokens or group.split("-")[0] in tokens


def strings(node):
    if isinstance(node, str):
        return [node]
    items = node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
    return [text for item in items for text in strings(item)]


def test_dead_end_seattle_25_houston_20():
    assert rule("ibc.1020.5", "seattle")["value"] == 25
    assert rule("ibc.1020.5", "houston")["value"] == 20
    assert rule("ibc.1020.5", "philadelphia")["value"] == 20
    assert rule("ibc.1020.5", "houston", occupancy="B", sprinklers="NFPA 13")["value"] == 50


def test_seattle_dead_end_exceptions():
    assert rule("ibc.1020.5", "seattle", occupancy="B", sprinklers="NFPA 13")["value"] == 50
    assert rule("ibc.1020.5", "seattle", occupancy="B", construction="IB", occupant_load=50)["value"] == 75
    assert rule("ibc.1020.5", "seattle", occupancy="B", construction="IA", occupant_load=50)["value"] == 75
    assert rule("ibc.1020.5", "seattle", occupancy="B", construction="IB", occupant_load=51)["value"] == 25
    assert rule("ibc.1020.5", "seattle", occupancy="B", construction="IIA", occupant_load=50)["value"] == 25
    assert rule("ibc.1020.5", "houston", occupancy="B", construction="IB", occupant_load=50)["value"] == 20


def test_dead_end_50_ft_groups():
    for group in GROUPS:
        reduced = group in "B E F-1 F-2 I-1 M R-1 R-2 S-1 S-2 U".split()
        assert rule("ibc.1020.5", "houston", occupancy=group, sprinklers="NFPA 13")["value"] == (50 if reduced else 20), group
        assert rule("ibc.1020.5", "seattle", occupancy=group, sprinklers="NFPA 13")["value"] == (50 if reduced else 25), group


def test_travel_distance_b_sprinklered():
    row = rule("ibc.1017.2", "houston", occupancy="B", sprinklers="NFPA 13")
    assert row == {"value": 300, "unit": "ft", "section": "Table 1017.2", "adopted": JURISDICTIONS["houston"], "title": ROWS["ibc.1017.2"]["title"]}
    assert rule("ibc.1017.2", "houston", occupancy="B", sprinklers="no")["value"] == 200
    assert set(JURISDICTIONS) == {"houston", "philadelphia", "seattle"}


def test_unknown_sprinkler_uses_stricter():
    assert stricter_when_unknown("ibc.1017.2", "houston", occupancy="B")["value"] == 200
    assert stricter_when_unknown("ibc.1017.2", "houston", occupancy="I-1")["value"] == "NP"
    assert stricter_when_unknown("ibc.1020.5", "seattle", occupancy="B")["value"] == 25
    assert stricter_when_unknown("ibc.1005.3.1", "houston", occupancy="B", voice_alarm="yes")["value"] == 0.3   # more inches per person is stricter
    assert stricter_when_unknown("ibc.1005.3.2", "houston", occupancy="B", voice_alarm="yes")["value"] == 0.2
    assert stricter_when_unknown("ibc.1006.3.3", "houston", occupant_load=10)["value"] == 2
    for status in (None, "conditional", "not_stated", "yes"):
        assert stricter_when_unknown("ibc.1017.2", "houston", occupancy="B", sprinklers=status)["value"] == 200


def test_a_sprinklered_value_needs_the_standard_its_row_names():
    """Presence with no standard read, or a standard the row does not take, gives the value without sprinklers.
    The code takes 13R in 1005.3, Table 1006.3.4(1) and the travel rows of A, E, F-1, M, R, S-1 and I-1, and NFPA 13 alone elsewhere."""
    for status in ("yes", "NFPA 13R", "NFPA 13D", None):
        assert rule("ibc.1017.2", "houston", occupancy="B", sprinklers=status)["value"] == 200, status
        assert rule("ibc.1020.5", "houston", occupancy="B", sprinklers=status)["value"] == 20, status
        assert rule(T1, "seattle", story=5, sprinklers=status)["value"] == "NP", status   # item 7 names NFPA 13
    for status in ("yes", "NFPA 13D", None):
        assert rule("ibc.1017.2", "houston", occupancy="R-2", sprinklers=status)["value"] == 200, status
        assert rule(T1, "houston", story=3, sprinklers=status)["value"] == "NP", status
        assert rule("ibc.1005.3.2", "houston", **{**REDUCED, "sprinklers": status})["value"] == 0.2, status
    assert rule("ibc.1017.2", "houston", occupancy="R-2", sprinklers="NFPA 13R")["value"] == 250
    assert rule(T1, "houston", story=3, sprinklers="NFPA 13R")["value"] == {"max_dwelling_units": 4, "max_exit_access_travel": 125}
    assert rule("ibc.1005.3.2", "houston", **{**REDUCED, "sprinklers": "NFPA 13R"})["value"] == 0.15


def test_every_sprinklered_case_takes_nfpa_13():
    """So a statement read as 13R or 13D where it means NFPA 13 never reaches a case NFPA 13 would not."""
    for row in ROWS.values():
        for case in [*row["cases"], *(case for cases in row.get("overrides", {}).values() for case in cases)]:
            assert "NFPA 13" in case.get("when", {}).get("sprinklers", ["NFPA 13"]), row["section"]


def test_nfpa_13_allows_one_exit_wherever_a_system_does_at_the_lowest_travel_limit():
    """With no system read, findings compares travel with Table 1006.3.4(1) at NFPA 13, because for travel the lower limit is the stricter."""
    for jurisdiction in JURISDICTIONS:
        for story in range(1, 8):
            limits = [found["value"]["max_exit_access_travel"] for found in (rule(T1, jurisdiction, story=story, sprinklers=s) for s in SYSTEMS) if found["value"] != "NP"]
            lenient = rule(T1, jurisdiction, story=story, sprinklers="NFPA 13")["value"]
            assert not limits or lenient != "NP" and lenient["max_exit_access_travel"] == min(limits), (jurisdiction, story)


def test_a_case_that_needs_a_fact_nobody_read_comes_back_under_unless():
    """No stage reads a corridor's use, so the 72 in of stretcher traffic can apply to any business corridor."""
    found = rule("ibc.1020.3", "houston", occupancy="B", occupant_load=120)
    assert found["value"] == 44 and [case["value"] for case in found["unless"]] == [72] and found["unless"][0]["confirm"] is True
    assert rule("ibc.1020.3", "houston", occupancy="B", occupant_load=120, use="stretcher traffic")["value"] == 72
    assert "unless" not in rule("ibc.1020.3", "houston", occupancy="R-2", occupant_load=120)
    assert "unless" not in rule("ibc.1017.2", "houston", occupancy="B", sprinklers="NFPA 13")


def test_one_exit_for_other_occupancies_asks_for_its_travel_limit():
    """Table 1006.3.4(2) also caps exit access travel, which is measured only as a minimum, so a person confirms it."""
    found = rule(T2, "houston", occupancy="B", story=2)
    assert found["value"] == 29 and found["confirm"] is True and found["conditions"] == ["75 ft exit access travel at most"]
    assert rule(T2, "houston", occupancy="H-3", story=1)["conditions"] == ["25 ft exit access travel at most"]
    assert "confirm" not in rule(T2, "houston", occupancy="B", story=3)


def test_every_row_has_section_value_unit():
    assert set(ROWS) >= {"ibc.1006.3.3", T1, "ibc.1017.2", "ibc.1020.3", "ibc.1023.3"}
    for row in ROWS.values():
        assert row["section"] and row["unit"] and row["title"] and row["cases"]
        for case in all_cases(row):
            assert case["value"] not in (None, "")


def test_no_row_contains_a_sentence_over_12_words():
    for row in ROWS.values():
        for text in strings(row):
            for sentence in re.split(r"[.;]\s", text):
                assert len(sentence.split()) <= 12, sentence


def test_seattle_r2_single_exit_differs_from_houston():
    houston = rule(T1, "houston", story=5, sprinklers="NFPA 13")
    seattle = rule(T1, "seattle", story=5, sprinklers="NFPA 13")
    assert houston["value"] == "NP" and "confirm" not in houston
    assert seattle["value"] == {"max_dwelling_units": 4, "max_exit_access_travel": 125}
    assert seattle["section"] == "1006.3.4 item 7"
    assert seattle["confirm"] is True and seattle["conditions"]


def test_seattle_single_exit_limits():
    assert rule(T1, "seattle", story=6, sprinklers="NFPA 13")["value"] == "NP"
    assert rule(T1, "seattle", story=5, sprinklers="no")["value"] == "NP"
    assert rule(T1, "seattle", sprinklers="NFPA 13")["value"] == "NP"
    assert stricter_when_unknown(T1, "seattle", story=5)["value"] == "NP"
    assert rule(T1, "seattle", story=3, sprinklers="NFPA 13") == {**rule(T1, "houston", story=3, sprinklers="NFPA 13"), "adopted": JURISDICTIONS["seattle"]}
    assert rule(T1, "seattle", story=4, sprinklers="NFPA 13")["confirm"] is True
    assert rule(T1, "seattle", story=4, sprinklers="NFPA 13")["section"] == "1006.3.4 item 7"


def test_r2_single_exit_asks_a_person():
    for jurisdiction in JURISDICTIONS:
        found = rule(T1, jurisdiction, story=3, sprinklers="NFPA 13")
        assert found["value"] == {"max_dwelling_units": 4, "max_exit_access_travel": 125}
        assert found["section"] == "Table 1006.3.4(1)"
        assert found["confirm"] is True
        assert found["conditions"] == ["4 dwelling units per floor at most", "Emergency escape openings at sleeping rooms"]
    assert rule(T1, "houston", story=4, sprinklers="NFPA 13")["value"] == "NP"


def test_exits_by_occupant_load():
    assert [rule("ibc.1006.3.3", "seattle", occupant_load=n)["value"] for n in (1, 500, 501, 1000, 1001)] == [2, 2, 3, 3, 4]


@pytest.mark.parametrize("groups, unsprinklered, sprinklered", [
    ("A-1 A-2 A-3 A-4 A-5 E F-1 M R-1 R-2 R-3 R-4 S-1", 200, 250),
    ("I-1", "NP", 250),
    ("B", 200, 300),
    ("F-2 S-2 U", 300, 400),
    ("H-1", "NP", 75),
    ("H-2", "NP", 100),
    ("H-3", "NP", 150),
    ("H-4", "NP", 175),
    ("H-5", "NP", 200),
    ("I-2 I-3", "NP", 200),
    ("I-4", 150, 200),
])
def test_travel_distance_table(groups, unsprinklered, sprinklered):
    for group in groups.split():
        assert rule("ibc.1017.2", "houston", occupancy=group, sprinklers="no")["value"] == unsprinklered, group
        assert rule("ibc.1017.2", "houston", occupancy=group, sprinklers="NFPA 13")["value"] == sprinklered, group


@pytest.mark.parametrize("groups, first, second", [
    ("A-1 A-2 A-3 A-4 A-5 E U", 49, "NP"),
    ("B F-1 F-2 M", 49, 29),
    ("H-2 H-3", 3, "NP"),
    ("H-4 H-5 I-1 I-2 I-3 I-4 R-1", 10, "NP"),
    ("S-1 S-2", 29, 29),
    ("H-1 R-2 R-3 R-4", "NP", "NP"),
])
def test_single_exit_table(groups, first, second):
    for group in groups.split():
        assert [rule(T2, "houston", occupancy=group, story=n)["value"] for n in (1, 2, 3)] == [first, second, "NP"], group


@pytest.mark.parametrize("groups, loads, widths", [
    ("A-1 A-2 A-3 A-4 A-5 B F-1 F-2 H-1 H-2 H-3 H-4 H-5 I-1 I-3 I-4 M R-1 R-2 R-3 R-4 S-1 S-2 U", (49, 50, 100, None), [36, 44, 44, 44]),
    ("E", (49, 50, 99, 100), [36, 44, 44, 72]),
    ("I-2", (49, 50, None), [96, 96, 96]),
])
def test_corridor_width_table(groups, loads, widths):
    for group in groups.split():
        assert [rule("ibc.1020.3", "houston", occupancy=group, occupant_load=n)["value"] for n in loads] == widths, group


def test_door_width_general():
    for group in set(GROUPS) - {"I-2"}:
        assert rule("ibc.1010.1.1", "houston", occupancy=group)["value"] == 32, group


def test_mixed_occupancy_fits_no_group():
    assert rule("ibc.1005.3.2", "houston", occupancy="A-3/I-2", sprinklers="NFPA 13", voice_alarm="yes")["value"] == 0.2
    assert rule(T2, "houston", occupancy="A-3/I-4", story=1)["value"] == "NP"


def test_number_given_as_text_counts_as_unread():
    assert rule(T2, "houston", occupancy="B", story="1")["value"] == "NP"
    assert rule("ibc.1020.3", "houston", occupancy="B", occupant_load="43")["value"] == 44


def test_occupancy_division_falls_in_its_group():
    assert rule("ibc.1017.2", "houston", occupancy="R-2", sprinklers="NFPA 13")["value"] == 250
    assert rule("ibc.1017.2", "houston", occupancy="S-2", sprinklers="NFPA 13")["value"] == 400
    assert rule("ibc.1017.2", "houston", occupancy="I-2", sprinklers="NFPA 13")["value"] == 200
    assert rule("ibc.1020.5", "houston", occupancy="R-3", sprinklers="NFPA 13")["value"] == 20
    assert rule(T2, "houston", occupancy="H-1", story=1)["value"] == "NP"


def test_reduced_exit_factor_needs_every_condition():
    for missing in ("occupancy", "sprinklers", "voice_alarm"):
        assert rule("ibc.1005.3.2", "houston", **{k: v for k, v in REDUCED.items() if k != missing})["value"] == 0.2
    for group in GROUPS:
        excluded = group == "I-2" or group.startswith("H")
        assert rule("ibc.1005.3.1", "houston", **{**REDUCED, "occupancy": group})["value"] == (0.3 if excluded else 0.2), group
        assert rule("ibc.1005.3.2", "houston", **{**REDUCED, "occupancy": group})["value"] == (0.2 if excluded else 0.15), group


def test_bed_movement_values_ask_a_person():
    for id, value in (("ibc.1020.3", 96), ("ibc.1010.1.1", 41.5)):
        found = rule(id, "houston", occupancy="I-2")
        assert found["value"] == value and found["confirm"] is True and found["conditions"]


@pytest.mark.parametrize("id, jurisdiction, inputs", [
    ("ibc.9999", "houston", {}),
    ("ibc.1023.3", "gotham", {}),
    ("ibc.1017.2", "houston", {"occupancy": "Business", "sprinklers": "NFPA 13"}),
    ("ibc.1017.2", "houston", {"sprinklers": "NFPA 13"}),
    ("ibc.1006.3.3", "houston", {"occupant_load": 0}),
    ("ibc.1017.2", "houston", {"occupancy": "A-3/B", "sprinklers": "NFPA 13"}),
    ("ibc.1017.2", "houston", {"occupancy": "A-3, B", "sprinklers": "NFPA 13"}),
    ("ibc.1006.3.3", "houston", {"occupant_load": "43"}),
    ("ibc.1006.3.3", "houston", {"occupant_load": "not printed"}),
    ("ibc.1020.3", "houston", {"occupancy": "E"}),
    ("ibc.1020.3", "houston", {"occupancy": "E", "occupant_load": "43"}),
    ("ibc.1020.3", "houston", {"occupancy": "not printed", "occupant_load": 150}),
    ("ibc.1020.3", "houston", {"occupant_load": 30}),
    ("ibc.1020.3", "houston", {"occupancy": "", "occupant_load": 30}),
    ("ibc.1020.3", "houston", {"occupancy": "I", "occupant_load": 30}),
    ("ibc.1010.1.1", "houston", {}),
    ("ibc.1010.1.1", "houston", {"occupancy": "not printed"}),
    ("ibc.1010.1.1", "houston", {"occupancy": "I"}),
])
def test_no_row_raises(id, jurisdiction, inputs):
    with pytest.raises(KeyError):
        rule(id, jurisdiction, **inputs)


@pytest.mark.skipif(not RESEARCH.exists(), reason="the research file stays out of git")
def test_values_match_research_file():
    research = {row["id"]: row for row in json.loads(RESEARCH.read_text())}
    for id, row in ROWS.items():
        assert (row["section"], row["title"]) == (research[id]["section"], research[id]["title"])
        assert row["unit"] in research[id]["units"]
    for id, index, field, inputs in PROBES:
        assert rule(id, "houston", **inputs)["value"] == research[id]["table"][index][field], (id, index)
    first = research[T1]["table"][0]
    assert rule(T1, "houston", story=3, sprinklers="NFPA 13")["value"] == {k: first[k] for k in ("max_dwelling_units", "max_exit_access_travel")}
    assert rule("ibc.1023.3", "houston")["value"] == research["ibc.1023.3"]["value"]
    single = research[T2]["table"]
    dead_end_groups = research["ibc.1020.5"]["table"][2]["case"].removesuffix(" with NFPA 13")
    for group in GROUPS:
        travel = next(row for row in research["ibc.1017.2"]["table"] if listed(row["occupancy"], group))
        assert rule("ibc.1017.2", "houston", occupancy=group, sprinklers="NFPA 13")["value"] == travel["sprinklered"], group
        assert rule("ibc.1017.2", "houston", occupancy=group, sprinklers="NFPA 13R")["value"] == travel["sprinklered" if "13R" in travel["sprinkler_standard"] else "unsprinklered"], group
        assert rule("ibc.1017.2", "houston", occupancy=group, sprinklers="no")["value"] == travel["unsprinklered"], group
        first = next((row for row in single[:4] if listed(row["occupancy"], group)), None)   # R-2 sleeping units are left out, no stage reads them
        found = rule(T2, "houston", occupancy=group, story=1)
        assert found["value"] == (first["max_occupants"] if first else "NP"), group
        assert found.get("conditions") == ([f"{first['max_exit_access_travel']} ft exit access travel at most"] if first else None), group
        assert rule(T2, "houston", occupancy=group, story=2)["value"] == (single[4]["max_occupants"] if listed(single[4]["occupancy"], group) else "NP"), group
        assert rule(T2, "houston", occupancy=group, story=3)["value"] == single[5]["max_occupants"], group
        for jurisdiction, general in (("houston", research["ibc.1020.5"]["table"][0]["max_length"]), ("seattle", 25)):
            expected = research["ibc.1020.5"]["table"][2]["max_length"] if listed(dead_end_groups, group) else general
            assert rule("ibc.1020.5", jurisdiction, occupancy=group, sprinklers="NFPA 13")["value"] == expected, group
            assert rule("ibc.1020.5", jurisdiction, occupancy=group, sprinklers="NFPA 13R")["value"] == general, group   # the research notes say 13R does not qualify
