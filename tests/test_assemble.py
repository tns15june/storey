"""review.json and the routes around it, on a hand-built review folder, so no test here runs a model."""
import json
import re
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from storey import meta, pipeline, server
from storey.assemble import _carried, ALL_DROPPED, CROP_MAX, CROP_PAD, OTHER_TRADE, assemble
from storey.letter import NOT_EVALUATED
from storey.pipeline import read_json, run, write_json

RID, CHILD = "a" * 32, "b" * 32
LATER = pipeline.STAGES[-4:]   # evaluate to assemble, the stages that make no model call on a first review
KEY = {"x-storey-key": "test-key"}
SIZES = {0: (4000, 3000), 1: (1000, 800), 2: (4000, 3000), 3: (3000, 2000), 4: (3000, 2000), 5: (3000, 2000)}   # factors 0.5, 2.0, 0.5, then 2/3 across and 1333/2000 down, because a JPEG has whole pixels
DOWN = 1333 / 2000
MINIMUM = "Minimum. Examiner to confirm."
STATUS = re.compile(r"(?i)\bpass(?:es|ed|ing)?\b|approv|compliant|compliance")


def row(i, number, title, cls="floor_plan", level="", read_ok=True):
    return {"index": i, "number": number, "title": title, "cls": cls, "level": level, "scale": "1/4 in", "plan_box": [0, 0, *SIZES[i]], "read_ok": read_ok}


def cell(state, point, box=None, mask=None, crop=None):
    return {"state": state, "signals": {"spark": "3/3", "crop": "yes", "ink": 0.9, "aligned": True, "reason": "3 of 3 runs"}, "point": point, "box": box, "crop": crop, "mask": mask}


def found(i):
    return cell("found", [800.0, 600.0], [700.0, 500.0, 900.0, 700.0], [[700.0, 500.0], [900.0, 500.0], [900.0, 700.0]], f"verticals/A_{i}.jpg")


READ = {"sheets": [row(0, "A1.1", "FIRST FLOOR PLAN", level="FIRST FLOOR"), row(1, "A1.2", "SECOND FLOOR PLAN", level="SECOND FLOOR"), row(2, "A1.3", "THIRD FLOOR PLAN", level="THIRD FLOOR"),
                   row(3, "A6.1", "DOOR SCHEDULE", "schedule"), row(4, "E1.1", "LIGHTING PLAN", "other"), row(5, "", "", "other", read_ok=False)]}
FLOORS = {"order": [0, 1, 2], "reference": 0, "transforms": {}, "inliers": {}, "unaligned": [], "stack": "stack.mp4", "frame_size": [960, 720], "discharge": 0, "not_checked": None}
VERTICALS = {"verticals": [{"id": "A", "name": "Main stair", "established": True, "pos": [800.0, 600.0],
                            "floors": {"0": found(0), "1": cell("missing", [400.0, 300.0], crop="verticals/A_1.jpg"), "2": found(2)}}], "not_checked": ["Elevators and shafts are not followed."]}
DECLARED = {"code": {"edition": {"value": "2021 International Building Code", "sheet": 3, "box": [300, 300, 600, 400], "reads": 2}, "occupancy": {"value": "B", "sheet": 3, "box": None, "reads": 2, "mentions": []}},
            "index": [], "titles": [], "tags": [{"sheet": 1, "kind": "door", "mark": "7", "point": [300, 200]}],
            "schedules": [{"sheet": 3, "kind": "door", "rows": [{"mark": "8", "cells": {}, "box": [100, 100, 900, 120]}], "source": "text", "plans": [1]}],
            "tables": [], "rooms": [], "exits": {"sheets": []}, "cannot_tell": []}
VECTOR = {"sheets": {"0": {"scale_in_per_pt": 0.667, "scale_checked": 12, "doors": [], "corridors": [], "skipped": None,
                           "routes": [{"path": [[100, 100], [100, 900]], "length_ft": 88.5, "from_room": "Office", "to_exit": "Stair A", "end_source": "matrix", "label": MINIMUM}]},
                     "3": {"scale_in_per_pt": None, "scale_checked": 0, "doors": [], "routes": [], "corridors": [], "skipped": "Not a floor plan."}}}
FACTS = {"read": READ, "floors": FLOORS, "verticals": VERTICALS, "declared": DECLARED, "vector": VECTOR}
TAGGED = DECLARED | {"tags": [*DECLARED["tags"], {"sheet": 1, "kind": "door", "mark": "8", "point": [500, 200]}]}   # door 8 gets its tag, so A-01, the first finding, leaves


def build(path, facts=FACTS):
    path.mkdir(parents=True)
    sheets = [{"index": i, "image": f"sheet_{i:03d}.png", "width": w, "height": h, "kind": "vector" if i < 3 else "scan", "text": "", "words": []} for i, (w, h) in SIZES.items()]
    for s in sheets:
        image = np.full((s["height"], s["width"], 3), 255, np.uint8)
        image[250:350, 350:450] = 0
        cv2.imwrite(str(path / s["image"]), image)
    write_json(path / "sheets.json", sheets)
    write_json(path / "status.json", {"state": "running", "stages": [], "usd": 0})
    for name, result in facts.items():
        write_json(path / f"stage_{name}.json", result)
    return path


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("a test reached the real client"))
    meta.set_transport(lambda kind, **request: {"text": None})   # an empty reply, so the letter keeps its template wording
    yield
    meta.set_transport(None)


@pytest.fixture
def review(tmp_path):
    path = build(tmp_path / "reviews" / RID)
    run(path, stages=LATER)
    return path


@pytest.fixture
def client(review, monkeypatch):
    monkeypatch.setattr(server, "DATA", review.parent)
    monkeypatch.setenv("STOREY_ACCESS_KEY", "test-key")
    return TestClient(server.app, base_url="http://127.0.0.1", headers=KEY)


def stored(review):
    return read_json(review / "review.json")


def one(items, **fields):
    return next(item for item in items if all(item.get(k) == v for k, v in fields.items()))


def test_stages_are_registered_in_order():
    assert [name for name, _, _ in pipeline.STAGES] == ["read", "floors", "verticals", "declared", "vector", "evaluate", "letter", "recheck", "assemble"]


def test_review_json_matches_contract(review):
    r = stored(review)
    assert r["id"] == RID
    assert r["project"] == {"name": "", "address": "", "cycle": 1, "code": "Houston Building Code 2021", "note": "", "jurisdiction": "houston", "adopted_code": "Houston Building Code 2021", "sprinklers": None, "discharge_level": "0",
                            "facts": {"edition": {"value": "2021 International Building Code", "sheet": "3", "box": [200.0, round(300 * DOWN, 1), 400.0, round(400 * DOWN, 1)], "reads": 2}, "occupancy": {"value": "B", "sheet": "3", "box": None, "reads": 2}}}
    assert r["floor_order"] == ["0", "1", "2"]
    assert r["sheets"][0] == {"id": "0", "number": "A1.1", "level": "FIRST FLOOR", "title": "FIRST FLOOR PLAN", "image": f"/api/reviews/{RID}/files/sheet_000.jpg", "w": 2000, "h": 1500, "class": "floor_plan", "kind": "vector",
                              "scale": "1/4 in", "reviewed": True, "reason": None}
    stair = r["verticals"][0]
    assert (stair["id"], stair["established"]) == ("A", True)
    assert stair["floors"]["1"]["state"] == "missing"
    assert stair["floors"]["0"]["signals"]["spark"] == "3/3"
    gap = one(r["findings"], check=9)
    assert set(gap) >= {"id", "key", "comment_id", "check", "discipline", "state", "severity", "sheet", "box", "title", "measured", "required", "rule", "evidence", "vertical", "crop"}
    assert gap["id"] == gap["comment_id"] and re.fullmatch(r"A-\d\d", gap["id"]) and gap["state"] is None and gap["vertical"] == "A"
    assert r["routes"] == [{"sheet": "0", "path": [[50.0, 50.0], [50.0, 450.0]], "length": 88.5, "label": MINIMUM}]
    assert one(r["open_items"], check=12)["vertical"] == "A"
    assert "Elevators and shafts are not followed." in r["not_checked"]
    assert r["cost"]["stages"] and r["letter"]["items"] and r["decisions"] == {}


def test_a_dwelling_draws_no_route_because_travel_distance_is_not_checked_there(tmp_path):
    residential = DECLARED | {"code": {"edition": {"value": "2015 Michigan Residential Code", "sheet": 3, "box": None, "reads": 2}}}
    path = build(tmp_path / "reviews" / RID, FACTS | {"declared": residential})
    run(path, stages=LATER)
    r = stored(path)
    assert any("travel distance" in line for line in r["not_checked"])
    assert r["routes"] == []


def test_sheet_ids_are_strings_so_the_screen_can_match_them(review):
    r = stored(review)
    ids = {s["id"] for s in r["sheets"]}
    assert all(type(i) is str for i in ids)
    assert all(f["sheet"] is None or f["sheet"] in ids for f in r["findings"])
    assert set(r["verticals"][0]["floors"]) <= ids


def test_sheet_images_are_jpegs_with_a_long_edge_of_2000(review):
    r = stored(review)
    for s in r["sheets"]:
        with Image.open(review / s["image"].rsplit("/", 1)[1]) as image:
            assert image.format == "JPEG" and image.size == (s["w"], s["h"])
        assert max(s["w"], s["h"]) == 2000


def test_coordinates_are_in_the_pixels_of_each_sheets_jpeg(review):
    r = stored(review)
    gap = one(r["findings"], check=9)   # on sheet 1, a 1000 px sheet that grows by 2
    assert gap["sheet"] == "1" and gap["box"] == [400.0, 200.0, 1200.0, 1000.0]
    tags = one(r["findings"], check=7, sheet="1")
    assert tags["box"] == [480.0, 280.0, 720.0, 520.0]
    rows = one(r["findings"], check=7, sheet="3")   # a 3000 px sheet that shrinks to 2000
    assert rows["box"] == [66.7, round(100 * DOWN, 1), 600.0, round(120 * DOWN, 1)]
    stair = r["verticals"][0]
    assert stair["pos"] == [400.0, 300.0]   # in the reference sheet
    assert stair["floors"]["0"]["point"] == [400.0, 300.0]
    assert stair["floors"]["0"]["box"] == [350.0, 250.0, 450.0, 350.0]
    assert stair["floors"]["0"]["mask"] == [[350.0, 250.0], [450.0, 250.0], [450.0, 350.0]]
    assert stair["floors"]["1"]["point"] == [800.0, 600.0] and stair["floors"]["1"]["box"] is None and stair["floors"]["1"]["mask"] is None
    top = one(r["open_items"], title="Stair A runs from FIRST FLOOR to THIRD FLOOR")
    assert top["sheet"] == "2" and top["box"] == [350.0, 250.0, 450.0, 350.0]


def test_crops_are_urls_the_screen_can_load(review):
    r = stored(review)
    assert r["verticals"][0]["floors"]["0"]["crop"] == f"/api/reviews/{RID}/files/verticals/A_0.jpg"


def test_each_finding_with_a_box_carries_a_crop_cut_from_the_full_sheet(review):
    r = stored(review)
    gap = one(r["findings"], check=9)
    assert gap["crop"] == f"/api/reviews/{RID}/files/crop_{gap['comment_id']}.jpg"
    crop = cv2.imread(str(review / f"crop_{gap['comment_id']}.jpg"))
    assert crop.shape[:2] == (500 + CROP_PAD, 400 + 2 * CROP_PAD)   # the box [200, 100, 600, 500] padded, and cut off by the top of the sheet
    assert crop.min() < 80   # the dark block at 350 to 450, 250 to 350 is inside it
    assert one(r["findings"], title="This sheet could not be read")["crop"] is None   # no box


def test_a_crop_over_the_limit_is_scaled_down(tmp_path):
    whole = DECLARED | {"schedules": [DECLARED["schedules"][0] | {"rows": [{"mark": "8", "cells": {}, "box": [0, 0, 3000, 2000]}]}]}
    path = build(tmp_path / "reviews" / RID, FACTS | {"declared": whole})
    run(path, stages=LATER)
    rows = one(stored(path)["findings"], check=7, sheet="3")
    assert max(cv2.imread(str(path / f"crop_{rows['id']}.jpg")).shape[:2]) == CROP_MAX


def test_letter_has_comment_ids_sheet_numbers_and_not_checked(review):
    letter = stored(review)["letter"]
    assert all(item["comment_id"] for item in letter["items"])
    assert {"A1.2", "A6.1"} <= {item["sheet"] for item in letter["items"]}   # headings from the read stage, not Page N
    assert "Elevators and shafts are not followed." in letter["not_checked"]["items"]   # evaluate's list


def test_same_choices_keep_the_stored_letter(review):
    letter = read_json(review / "stage_letter.json")
    letter["items"][0]["text"] = "Reworded by the model."
    write_json(review / "stage_letter.json", letter)
    assert assemble(review)["letter"]["items"][0]["text"] == "Reworded by the model."


def test_a_selector_round_trip_brings_the_models_wording_back(review):
    letter = read_json(review / "stage_letter.json")
    letter["items"][0]["text"] = "Reworded by the model."
    write_json(review / "stage_letter.json", letter)
    for choices in (("seattle",), ("seattle",), ("seattle", "no")):   # a Keep click repeats the selection, and a second change must not forget what the stage ran under
        assert assemble(review, *choices)["letter"]["items"][0]["text"] != "Reworded by the model."
    assert read_json(review / "stage_letter.json") == letter
    assert assemble(review)["letter"]["items"][0]["text"] == "Reworded by the model."


def test_comment_ids_survive_a_change_of_jurisdiction_and_back(review):
    before = {f["key"]: f["comment_id"] for f in stored(review)["findings"]}
    write_json(review / "stage_verticals.json", VERTICALS | {"verticals": []})   # the stair findings leave under the next selection
    gone = assemble(review, "seattle")
    assert not any(f["check"] == 9 for f in gone["findings"])
    write_json(review / "stage_verticals.json", VERTICALS)
    back = assemble(review)
    assert {f["key"]: f["comment_id"] for f in back["findings"]} == before


def test_a_failed_stage_still_gives_a_review_that_says_so(tmp_path):
    path = build(tmp_path / "reviews" / RID, FACTS | {"verticals": {"failed": "This review reached its spend cap of 4.00 USD."}})
    run(path, stages=LATER)
    r = stored(path)
    assert r["verticals"] == [] and r["sheets"]
    assert any("spend cap of 4.00 USD" in line for line in r["not_checked"])


def test_a_failed_evaluate_still_gives_a_review_that_says_so(tmp_path):
    path = build(tmp_path / "reviews" / RID, FACTS | {"read": {"failed": "Read the sheets did not finish."}, "evaluate": {"failed": "Run the checks did not finish."},
                                                      "letter": {"header": {}, "items": [], "note": NOT_EVALUATED, "not_checked": {"line": "", "items": []}}})
    r = assemble(path)
    assert r["findings"] == [] and r["open_items"] == []
    assert r["letter"]["note"] == NOT_EVALUATED
    assert any("Run the checks did not finish." in line for line in r["not_checked"])
    assert any("Read the sheets did not finish." in line for line in r["not_checked"])   # evaluate never reported this one
    assert [s["reviewed"] for s in r["sheets"]] == [False] * 6


def broken(ctx):
    raise RuntimeError("stage stopped")


def failed_review(tmp_path, *names):
    """A review through the real later stages, with each stage in `names` raising. With no floors the stair stage stops before any model call, so it runs too."""
    gone = {*names, *(("verticals",) if "floors" in names else ())}
    path = build(tmp_path / "reviews" / RID, {name: result for name, result in FACTS.items() if name not in gone})
    run(path, stages=[(name, label, broken if name in names else fn) for name, label, fn in pipeline.STAGES])
    return stored(path)


@pytest.mark.parametrize("stage", ["floors", "verticals", "declared", "evaluate"])
def test_a_stage_that_did_not_finish_leaves_no_sheet_reviewed(tmp_path, stage):
    sheets = failed_review(tmp_path, stage)["sheets"]
    assert [s["reviewed"] for s in sheets] == [False] * 6
    assert all(stage in s["reason"] for s in sheets[:4])
    assert sheets[4]["reason"] == OTHER_TRADE and "could not be read" in sheets[5]["reason"]


def test_a_stage_with_no_result_counts_as_not_finished(tmp_path):
    path = build(tmp_path / "reviews" / RID, {name: result for name, result in FACTS.items() if name != "vector"})
    run(path, stages=LATER)
    assert "vector" in stored(path)["sheets"][0]["reason"]   # evaluate reads it the same way


def test_every_stage_that_did_not_finish_is_named_on_the_sheet(tmp_path):
    reason = failed_review(tmp_path, "verticals", "declared")["sheets"][0]["reason"]
    assert "verticals" in reason and "declared" in reason


@pytest.mark.parametrize("stage, drawn", [("floors", 0), ("declared", 0), ("evaluate", 0), ("verticals", 1)])
def test_a_route_is_drawn_only_where_travel_distance_was_checked(tmp_path, stage, drawn):
    assert len(failed_review(tmp_path, stage)["routes"]) == drawn


def test_a_plan_outside_the_floor_order_says_what_was_not_checked(tmp_path):
    path = build(tmp_path / "reviews" / RID, FACTS | {"floors": FLOORS | {"order": [0, 2]}})
    run(path, stages=LATER)
    r = stored(path)
    sheets = {s["id"]: s for s in r["sheets"]}
    assert sheets["1"]["reviewed"] is False and "stairs, travel distance and corridor width" in sheets["1"]["reason"]
    assert (sheets["0"]["reviewed"], sheets["0"]["reason"]) == (True, None)
    assert one(r["findings"], title="Stairs, travel distance and corridor width were not checked on A1.2")


def test_a_copy_of_a_floor_left_out_on_purpose_stays_reviewed(tmp_path):
    copy = READ | {"sheets": [*READ["sheets"][:3], row(3, "A6.1", "FIRST FLOOR LIFE SAFETY PLAN", level="FIRST FLOOR"), *READ["sheets"][4:]]}
    path = build(tmp_path / "reviews" / RID, FACTS | {"read": copy})
    run(path, stages=LATER)
    assert stored(path)["sheets"][3]["reviewed"] is True


def test_a_failed_letter_says_so(tmp_path):
    path = build(tmp_path / "reviews" / RID, FACTS | {"evaluate": {"findings": [], "open_items": [], "not_checked": []}, "letter": {"failed": "Draft the comment letter did not finish."}})
    assert any("Draft the comment letter did not finish." in line for line in assemble(path)["not_checked"])


def test_sheets_outside_the_review_say_why(review):
    sheets = {s["id"]: s for s in stored(review)["sheets"]}
    assert (sheets["3"]["reviewed"], sheets["3"]["reason"]) == (True, None)
    assert sheets["4"]["reviewed"] is False and "architectural" in sheets["4"]["reason"]
    assert sheets["5"]["reviewed"] is False and "could not be read" in sheets["5"]["reason"]
    assert sheets["5"]["level"] == "Page 6"   # the screen prints level on every row


def test_a_trade_sheet_is_found_by_its_title_alone(tmp_path):
    riser = READ | {"sheets": [*READ["sheets"][:3], row(3, "A6.1", "PLUMBING RISER DIAGRAM", "other"), *READ["sheets"][4:]]}
    path = build(tmp_path / "reviews" / RID, FACTS | {"read": riser})
    run(path, stages=LATER)
    sheet = stored(path)["sheets"][3]
    assert (sheet["reviewed"], sheet["reason"]) == (False, OTHER_TRADE)


def test_status_words_never_reach_the_review(tmp_path):
    hostile = READ | {"sheets": [READ["sheets"][0] | {"title": "APPROVED BY THE CITY"}, *READ["sheets"][1:]]}
    path = build(tmp_path / "reviews" / RID, FACTS | {"read": hostile})
    write_json(path / "stage_upload.json", {"name": "Approved set", "parent": None})
    run(path, stages=LATER)
    assert not STATUS.search(json.dumps(stored(path)))


def test_a_second_assemble_makes_no_images(review, monkeypatch):
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("a sheet was decoded again"))
    assemble(review)


def test_cost_is_kept_per_stage_across_reruns(tmp_path):
    path = build(tmp_path / "reviews" / RID)
    write_json(path / "cost.json", {"ingest": {"usd": 0, "seconds": 3.0}})
    stages = [("spend", "Spend", lambda ctx: ctx.meter.add(0.25, model="m") or {}), *LATER]
    run(path, stages=stages)
    (path / "stage_assemble.json").unlink()
    run(path, stages=stages)
    costs = read_json(path / "cost.json")
    assert list(costs) == ["ingest", "spend", "evaluate", "letter", "recheck", "assemble"]
    assert costs["spend"]["usd"] == 0.25 and costs["spend"]["seconds"] >= 0
    r = stored(path)
    assert r["cost"]["usd"] == pytest.approx(0.25) and r["cost"]["seconds"] >= 3.0
    assert {"name": "ingest", "usd": 0, "seconds": 3.0} in r["cost"]["stages"]


def second_review(review, facts=FACTS, name=CHILD):
    child = build(review.parent / name, facts)
    write_json(child / "stage_upload.json", {"name": "Second set", "parent": review.name})
    run(child, stages=LATER)
    return child


def ids(review):
    return {f["key"]: f["comment_id"] for f in stored(review)["findings"]}


def test_a_second_review_keeps_its_parents_comment_ids(review):
    first = stored(review)["findings"][0]
    assert first["comment_id"] == "A-01"
    child = second_review(review, FACTS | {"declared": TAGGED})
    assert first["key"] not in ids(child)   # plain numbering would now shift every id down by one
    assert ids(child).items() < ids(review).items()
    second = stored(child)
    assert second["project"]["cycle"] == 2 and second["project"]["name"] == "Second set" and second["parent"] == RID


def test_a_second_review_keeps_its_parents_selections(review):
    assemble(review, "seattle", "no", 2)
    child = second_review(review, FACTS | {"floors": FLOORS | {"discharge": 1}})
    project = stored(child)["project"]
    assert (project["jurisdiction"], project["sprinklers"], project["discharge_level"]) == ("seattle", "no", "1")   # its own floors stage's level, because a sheet index does not carry across sets
    assert any("2021 Seattle Building Code" in item["text"] for item in stored(child)["letter"]["items"])


def test_a_second_review_keeps_its_parents_decisions(client, review):
    key = next(item["key"] for item in stored(review)["letter"]["items"])
    client.post(f"/api/reviews/{RID}/decisions", json={"key": key, "decision": "drop"})
    child = second_review(review)
    assert stored(child)["decisions"] == {key: "drop"}
    assert key not in [item["key"] for item in stored(child)["letter"]["items"]]
    third = second_review(child, name="c" * 32)   # cycle 3 reads cycle 2's review.json, so cycle 1's decision reaches it
    assert stored(third)["decisions"] == {key: "drop"} and stored(third)["project"]["cycle"] == 3
    body = client.post(f"/api/reviews/{CHILD}/decisions", json={"key": key, "decision": None}).json()
    assert key in [item["key"] for item in body["letter"]["items"]]


def test_a_drop_does_not_carry_to_a_finding_that_changed(client, review):
    key = next(f["key"] for f in stored(review)["findings"] if f["check"] == 9)
    client.post(f"/api/reviews/{RID}/decisions", json={"key": key, "decision": "drop"})
    changed = json.loads(json.dumps(VERTICALS))
    changed["verticals"][0]["floors"]["1"]["state"] = "check"
    child = second_review(review, FACTS | {"verticals": changed})
    assert key not in stored(child)["decisions"]
    assert key in [item["key"] for item in stored(child)["letter"]["items"]]


@pytest.mark.parametrize("field", ["title", "measured", "required", "severity", "rule"])
def test_a_drop_stays_only_on_a_finding_that_reads_the_same(field):
    finding = {"key": "k", "title": "t", "measured": "m", "required": "r", "severity": "check", "rule": "s"}
    earlier = {"findings": [finding], "decisions": {"k": "drop"}}
    assert _carried(earlier, [finding]) == {"k": "drop"}
    assert _carried(earlier, [finding | {field: "changed"}]) == {}


def test_a_rerun_of_the_evaluate_stage_keeps_its_ids(review):
    before = ids(review)
    write_json(review / "stage_declared.json", TAGGED)
    run(review, fresh=True, stages=LATER)
    assert ids(review).items() < before.items()


def test_the_assemble_stage_keeps_the_stored_choices(review):
    assemble(review, "seattle", "no", 1)
    (review / "stage_assemble.json").unlink()
    run(review, stages=pipeline.STAGES[-1:])
    project = stored(review)["project"]
    assert (project["jurisdiction"], project["sprinklers"], project["discharge_level"]) == ("seattle", "no", "1")


# The routes


def test_evaluate_makes_no_model_call(client, review):
    calls = []
    meta.set_transport(lambda kind, **request: calls.append(kind))
    r = client.post(f"/api/reviews/{RID}/evaluate", json={"jurisdiction": "seattle", "sprinklers": "yes", "discharge": 1})
    assert r.status_code == 200
    body = r.json()
    assert calls == []
    assert body == stored(review)
    assert (body["project"]["jurisdiction"], body["project"]["sprinklers"], body["project"]["discharge_level"]) == ("seattle", "yes", "1")
    assert body["project"]["code"] == body["project"]["adopted_code"] == "2021 Seattle Building Code"
    assert one(body["findings"], check=10)["sheet"] == "1"


def test_evaluate_rebuilds_the_letter(client, review):
    old = read_json(review / "stage_letter.json")
    letter = client.post(f"/api/reviews/{RID}/evaluate", json={"jurisdiction": "seattle", "discharge": 1}).json()["letter"]
    assert letter != old and read_json(review / "stage_letter.json") == old
    assert any("2021 Seattle Building Code" in item["text"] for item in letter["items"])
    assert all(item["comment_id"] for item in letter["items"])


def test_evaluate_keeps_ids_of_findings_that_stay(client, review):
    before = {f["key"]: f["id"] for f in stored(review)["findings"]}
    after = client.post(f"/api/reviews/{RID}/evaluate", json={"jurisdiction": "philadelphia"}).json()
    for f in after["findings"]:
        assert before.get(f["key"], f["id"]) == f["id"]


@pytest.mark.parametrize("body", [{"jurisdiction": "gotham"}, {"sprinklers": "maybe"}, {"discharge": 3}, {"discharge": "first"}])
def test_evaluate_refuses_a_bad_choice(client, body):
    assert client.post(f"/api/reviews/{RID}/evaluate", json=body).status_code == 422


def test_routes_refuse_a_review_still_running(client, review):
    write_json(review / "status.json", {"state": "running"})
    key = stored(review)["findings"][0]["key"]
    assert client.post(f"/api/reviews/{RID}/evaluate", json={}).status_code == 409
    assert client.post(f"/api/reviews/{RID}/decisions", json={"key": key, "decision": "keep"}).status_code == 409


def test_decision_is_saved(client, review):
    key = stored(review)["findings"][0]["key"]
    r = client.post(f"/api/reviews/{RID}/decisions", json={"key": key, "decision": "keep"})
    assert r.status_code == 200 and r.json()["decisions"] == {key: "keep"}
    assert read_json(review / "decisions.json") == {key: "keep"}


def test_dropped_finding_leaves_the_letter_and_comes_back(client, review):
    key = next(item["key"] for item in stored(review)["letter"]["items"])
    assert key not in [item["key"] for item in client.post(f"/api/reviews/{RID}/decisions", json={"key": key, "decision": "drop"}).json()["letter"]["items"]]
    assert key in [item["key"] for item in client.post(f"/api/reviews/{RID}/decisions", json={"key": key, "decision": None}).json()["letter"]["items"]]


def test_a_letter_with_every_comment_dropped_says_so(client, review):
    keys = [item["key"] for item in stored(review)["letter"]["items"]]
    for key in keys:
        letter = client.post(f"/api/reviews/{RID}/decisions", json={"key": key, "decision": "drop"}).json()["letter"]
    assert letter["items"] == [] and letter["note"] == ALL_DROPPED
    letter = client.post(f"/api/reviews/{RID}/decisions", json={"key": keys[0], "decision": None}).json()["letter"]
    assert letter["items"] and letter["note"] != ALL_DROPPED


def test_a_decision_outlives_a_change_of_jurisdiction(client, review):
    key = next(item["key"] for item in stored(review)["letter"]["items"])
    client.post(f"/api/reviews/{RID}/decisions", json={"key": key, "decision": "drop"})
    body = client.post(f"/api/reviews/{RID}/evaluate", json={"jurisdiction": "seattle"}).json()
    assert body["decisions"] == {key: "drop"}
    assert key not in [item["key"] for item in body["letter"]["items"]]


def test_a_decision_keeps_the_selected_jurisdiction(client, review):
    body = client.post(f"/api/reviews/{RID}/evaluate", json={"jurisdiction": "seattle", "sprinklers": "no"}).json()
    after = client.post(f"/api/reviews/{RID}/decisions", json={"key": body["findings"][0]["key"], "decision": "keep"}).json()
    assert (after["project"]["jurisdiction"], after["project"]["sprinklers"]) == ("seattle", "no")


@pytest.mark.parametrize("body, code", [({"key": "not-a-finding", "decision": "keep"}, 404), ({"key": "x", "decision": "approve"}, 422), ({"decision": "keep"}, 422)])
def test_a_bad_decision_is_refused_and_saves_nothing(client, review, body, code):
    assert client.post(f"/api/reviews/{RID}/decisions", json=body).status_code == code
    assert not (review / "decisions.json").exists()


def test_stair_crop_is_served_and_a_path_outside_the_folder_is_refused(client, review):
    (review / "verticals").mkdir()
    (review / "verticals" / "A_0.jpg").write_bytes(b"\xff\xd8 crop")
    (review / "upload").mkdir()
    (review / "upload" / "0000.pdf").write_bytes(b"%PDF")
    (review.parent / "secret.txt").write_text("not yours")
    assert client.get(f"/api/reviews/{RID}/files/verticals/A_0.jpg").content == b"\xff\xd8 crop"
    assert client.get(f"/api/reviews/{RID}/files/review.json").status_code == 200
    for name in ("verticals/..%2F..%2Fsecret.txt", "..%2Fsecret.txt", "upload/0000.pdf", "verticals/%2E%2E/%2E%2E/secret.txt", "verticals/nothing.jpg", "verticals"):
        assert client.get(f"/api/reviews/{RID}/files/{name}").status_code == 404, name


def test_past_reviews_are_listed_newest_first(client, review):
    write_json(review / "stage_upload.json", {"name": "THE-CAPITOL", "parent": None})
    other = build(review.parent / CHILD)
    write_json(other / "status.json", {"state": "failed", "sheets": 2, "usd": 1.5})
    (review.parent / "live_floors_center").mkdir()      # not a review id
    write_json(review.parent / "live_floors_center" / "status.json", {"state": "done"})
    (review.parent / "ledger.jsonl").write_text("")
    rows = client.get("/api/reviews").json()
    assert [row["id"] for row in rows] == [CHILD, RID]
    assert rows[0] | {"updated": 0} == {"id": CHILD, "name": "", "parent": None, "state": "failed", "sheets": 2, "usd": 1.5, "updated": 0}
    assert rows[1]["name"] == "THE-CAPITOL" and rows[1]["state"] == "done"


def test_an_upload_name_shows_no_status_word_on_the_list_of_reviews(client):
    rid = client.post("/api/reviews", files=[("files", ("Approved set.png", png(), "image/png"))]).json()["id"]
    finished(client, rid)
    assert next(row["name"] for row in client.get("/api/reviews").json() if row["id"] == rid) == "[word removed] set"


def test_a_long_upload_name_is_cut(client):
    rid = client.post("/api/reviews", files=[("files", ("x" * 200 + ".png", png(), "image/png"))]).json()["id"]
    finished(client, rid)
    assert next(row["name"] for row in client.get("/api/reviews").json() if row["id"] == rid) == "x" * server.MAX_NAME


def test_past_reviews_with_no_folder_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA", tmp_path / "none")
    monkeypatch.setenv("STOREY_ACCESS_KEY", "test-key")
    assert TestClient(server.app, base_url="http://127.0.0.1", headers=KEY).get("/api/reviews").json() == []


def finished(client, rid):
    for _ in range(1500):
        st = client.get(f"/api/reviews/{rid}").json()
        if st["state"] in ("done", "failed"):
            return st
        time.sleep(.02)
    raise AssertionError("review did not finish")


def png(side=300):
    return cv2.imencode(".png", np.full((side, side * 2, 3), 255, np.uint8))[1].tobytes()


def test_recheck_starts_a_new_review_with_parent_set(client, review):
    r = client.post(f"/api/reviews/{RID}/recheck", files=[("files", ("A1.png", png(), "image/png"))])
    assert r.status_code == 200
    rid = r.json()["id"]
    assert rid != RID and finished(client, rid)["state"] == "done"
    assert stored(review.parent / rid)["parent"] == RID


def test_recheck_refuses_a_review_still_running_and_saves_nothing(client, review):
    write_json(review / "status.json", {"state": "running"})
    assert client.post(f"/api/reviews/{RID}/recheck", files=[("files", ("A1.png", png(), "image/png"))]).status_code == 409
    assert [p.name for p in review.parent.iterdir() if p.is_dir()] == [RID]


def test_images_run_every_stage_to_a_review_when_every_reply_is_empty(client, review):
    """No sheet can be read, so nothing is known: the review still finishes and says so."""
    rid = client.post("/api/reviews", files=[("files", (f"{i}.png", png(), "image/png")) for i in range(3)]).json()["id"]
    st = finished(client, rid)
    assert st["state"] == "done" and st["sheets"] == 3
    assert [row["state"] for row in st["stages"]] == ["done"] * 9
    r = stored(review.parent / rid)
    assert r["project"]["name"] == "3 page images"
    assert [s["reviewed"] for s in r["sheets"]] == [False] * 3
    assert sum(f["title"] == "This sheet could not be read" for f in r["findings"]) == 3
    assert any("Stairs were not followed" in line for line in r["not_checked"])
    assert [s["name"] for s in r["cost"]["stages"]][:2] == ["ingest", "read"]
    assert client.get(f"/api/reviews/{rid}/files/sheet_002.jpg").status_code == 200
