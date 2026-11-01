"""The test tool on a hand-built three-floor review. Spark is stubbed, and the stubbed crop answer looks at the crop it receives, so a stair left on the sheet stays Found and fails these tests."""
import base64
import hashlib
import json
import re
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from storey import meta, pipeline, server, testtool
from storey.assemble import assemble
from storey.floors import ink
from storey.pipeline import read_json, run, write_json
from storey.testtool import remove_stair
from storey.verticals import FLOOR_P, INK_LOW, QUESTION

RID = "a" * 32
KEY = {"x-storey-key": "test-key"}
W, H = 3000, 2000
BOX = [1200.0, 800.0, 1500.0, 1100.0]   # sheet px round the stair
TREADS = (1220, 1480, range(820, 1090, 18))
IDENTITY = np.eye(3).tolist()


def sheet_image():
    """A stair room with walls 12 px thick and treads 2 px thin on a 3000 px sheet, so 8 and 1.3 px on the 2000 px copy the removal rule was tested on."""
    im = np.full((H, W, 3), 245, np.uint8)
    cv2.rectangle(im, (300, 300), (2700, 1700), (20, 20, 20), 12)
    cv2.rectangle(im, (1190, 790), (1510, 1110), (20, 20, 20), 12)
    for y in TREADS[2]:
        cv2.line(im, (TREADS[0], y), (TREADS[1], y), (20, 20, 20), 2)
    return im


def well_image():
    """The stair on habs_stelizabeths_center: a grey well 16 px wide on the 2000 px copy, with tread stubs either side of it, close enough to blend to grey at that size."""
    im = np.full((H, W, 3), 245, np.uint8)
    cv2.rectangle(im, (300, 300), (2700, 1700), (20, 20, 20), 12)
    cv2.rectangle(im, (1190, 790), (1510, 1110), (20, 20, 20), 12)
    for y in range(820, 1085, 9):
        cv2.line(im, (1215, y), (1485, y), (20, 20, 20), 2)
    cv2.rectangle(im, (1338, 850), (1362, 1050), (190, 190, 190), -1)
    cv2.rectangle(im, (1338, 850), (1362, 1050), (20, 20, 20), 3)
    return im


def cell(state="found", sheet=0):
    return {"state": state, "signals": {"spark": "3/3", "crop": "yes", "ink": 1.0, "aligned": True, "reason": "3 of 3 runs"}, "point": [1350.0, 950.0], "box": BOX, "crop": f"verticals/A_{sheet}.jpg", "mask": [[1200.0, 800.0], [1500.0, 800.0], [1500.0, 1100.0]]}


def build(path):
    path.mkdir(parents=True)
    sheets = [{"index": i, "image": f"sheet_{i:03d}.png", "width": W, "height": H, "kind": "scan", "text": "", "words": []} for i in range(3)]
    for s in sheets:
        cv2.imwrite(str(path / s["image"]), sheet_image())
    write_json(path / "sheets.json", sheets)
    (path / "stack.mp4").write_bytes(b"stack")
    (path / "upload").mkdir()
    (path / "upload" / "0000.pdf").write_bytes(b"%PDF original upload")
    write_json(path / "stage_upload.json", {"name": "Tower", "parent": None})
    write_json(path / "status.json", {"state": "done", "sheets": 3, "usd": 0.5})
    write_json(path / "cost.json", {"ingest": {"usd": 0, "seconds": 3.0}, "verticals": {"usd": 0.5, "seconds": 60.0}})
    levels = ("FIRST FLOOR", "SECOND FLOOR", "THIRD FLOOR")
    write_json(path / "stage_read.json", {"sheets": [{"index": i, "number": f"A{i + 1}", "title": f"{level} PLAN", "cls": "floor_plan", "level": level, "scale": "", "plan_box": [0, 0, W, H], "read_ok": True}
                                                     for i, level in enumerate(levels)]})
    write_json(path / "stage_floors.json", {"order": [0, 1, 2], "order_read": True, "reference": 0, "transforms": {str(i): IDENTITY for i in range(3)}, "inliers": {"0": 99, "1": 99, "2": 99}, "unaligned": [], "stack": "stack.mp4",
                                            "frame_size": [960, 640], "discharge": 0, "not_checked": None})
    write_json(path / "stage_verticals.json", {"verticals": [{"id": "A", "name": "Main stair", "established": True, "pos": [1350.0, 950.0], "floors": {str(i): cell(sheet=i) for i in range(3)}}], "not_checked": []})
    write_json(path / "stage_declared.json", {"code": {}, "index": [], "titles": [], "tags": [], "schedules": [], "tables": [], "rooms": [], "exits": {"sheets": []}, "cannot_tell": []})
    write_json(path / "stage_vector.json", {"sheets": {str(i): {"skipped": "A scan.", "routes": [], "doors": [], "corridors": [], "scale_in_per_pt": None, "scale_checked": 0} for i in range(3)}})
    return path


def compare_reply():
    """Every run claims the stair on every floor, so nothing but the evidence on the sheet can catch the removal."""
    return json.dumps({"stairs": [{"name": "Main stair", "floors": [{"floor": f"F{k + 1}", "present": True, "x": 450, "y": 475} for k in range(3)]}], "shafts": []})


def crop_answer(request):
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(request["messages"][-1]["content"][1]["image_url"]["url"].split(",", 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = crop.shape
    centre = crop[h // 3:2 * h // 3, w // 3:2 * w // 3]
    return json.dumps({"answer": "yes" if (centre < 128).mean() > 0.03 else "no", "seen": "lines"})


def fake(kind, **request):
    assert kind == "chat", "this building draws no outline, so SAM is not called"
    prompt = request["messages"][-1]["content"][0]["text"]
    if prompt.startswith(FLOOR_P[:40]):
        return {"text": compare_reply(), "input": 1000, "output": 500}
    if prompt == QUESTION:
        return {"text": crop_answer(request), "input": 100, "output": 50}
    if "object_name" in prompt:
        return {"text": json.dumps({"objects": []}), "input": 100, "output": 50}
    return {"text": None}   # the letter keeps its template wording


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("a test reached the real client"))
    meta.set_transport(fake)
    yield
    meta.set_transport(None)


@pytest.fixture
def review(tmp_path):
    path = build(tmp_path / "reviews" / RID)
    run(path)
    (path / "verticals").mkdir()
    (path / "verticals" / "A_1.jpg").write_bytes(b"\xff\xd8 old crop")
    return path


@pytest.fixture
def client(review, monkeypatch):
    monkeypatch.setattr(server, "DATA", review.parent)
    monkeypatch.setenv("STOREY_ACCESS_KEY", "test-key")
    return TestClient(server.app, base_url="http://127.0.0.1", headers=KEY)


@pytest.fixture
def well(review):
    """`review` with the grey well on every floor, and an outline that stops at the well, as SAM's did: 38 px of a stair about 75 px wide."""
    for i in range(3):
        cv2.imwrite(str(review / f"sheet_{i:03d}.png"), well_image())
    verticals = read_json(review / "stage_verticals.json")
    for c in verticals["verticals"][0]["floors"].values():
        c |= {"box": [1350.0, 800.0, 1500.0, 1100.0], "mask": None}
    write_json(review / "stage_verticals.json", verticals)
    return review


@pytest.fixture
def walled(review):
    """`review` with a solid wall through the stair's place on every floor, which a removal must keep and the ink test reads."""
    for i in range(3):
        sheet = cv2.imread(str(review / f"sheet_{i:03d}.png"))
        sheet[700:1200, 1330:1370] = (20, 20, 20)
        cv2.imwrite(str(review / f"sheet_{i:03d}.png"), sheet)
    return review


def fingerprint(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob("*")) if p.is_file()}


def folders(review):
    return sorted(p.name for p in review.parent.iterdir() if p.is_dir())


def finished(client, rid):
    for _ in range(1500):
        st = client.get(f"/api/reviews/{rid}").json()
        if st["state"] in ("done", "failed"):
            return st
        time.sleep(.02)
    raise AssertionError("review did not finish")


def test_remove_stair_leaves_original_untouched(review):
    before = fingerprint(review)
    copy = remove_stair(review, "A", 1)
    run(copy)
    assert copy != review and copy.parent == review.parent
    assert fingerprint(review) == before


def test_the_copy_reuses_finished_stages_except_verticals_evaluate_and_letter(review):
    copy = remove_stair(review, "A", 1)
    for name in ("read", "floors", "declared", "vector"):
        assert (copy / f"stage_{name}.json").read_bytes() == (review / f"stage_{name}.json").read_bytes()
    for name in ("verticals", "evaluate", "letter", "assemble"):
        assert not (copy / f"stage_{name}.json").exists()


def test_the_copy_holds_nothing_the_review_would_show_stale(review):
    copy = remove_stair(review, "A", 1)
    assert not list(copy.glob("sheet_*.jpg")) and not list(copy.glob("crop_*.jpg"))
    assert not (copy / "review.json").exists() and not (copy / "verticals").exists()


def test_the_copy_starts_its_own_spend_and_status(review):
    copy = remove_stair(review, "A", 1)
    status = read_json(copy / "status.json")
    assert (status["state"], status["usd"], status["sheets"]) == ("queued", 0, 3)
    assert not (copy / "cost.json").exists()


def test_removal_takes_the_treads_and_keeps_the_thick_walls(review):
    copy = remove_stair(review, "A", 1)
    old, new = (cv2.imread(str(p / "sheet_001.png")) for p in (review, copy))
    assert old.shape == new.shape == (H, W, 3)
    treads = new[820:1080, 1230:1470]
    assert (cv2.cvtColor(treads, cv2.COLOR_BGR2GRAY) < 128).mean() < 0.002
    assert (old[784:796, 1190:1510] == new[784:796, 1190:1510]).all(), "the wall on the box's edge stays, pixel for pixel"
    assert (old[:700] == new[:700]).all() and (old[1200:] == new[1200:]).all()


@pytest.mark.parametrize("wall", [(1400, 700, 1421, 1200), (1100, 1040, 1600, 1061)], ids=["upright", "level"])
def test_a_hatched_wall_stays_though_the_ink_map_sees_only_its_outlines(review, wall):
    """14 px wide on the 2000 px copy, clear of the ink test's place and on past the box, with hatching that blends to grey at that size, as on habs_stelizabeths_center."""
    x1, y1, x2, y2 = wall
    hatch = np.full((y2 - y1, x2 - x1, 3), 245, np.uint8)
    for c in range(y1 - y2, x2 - x1, 7):
        cv2.line(hatch, (c, 0), (c + y2 - y1, y2 - y1), (40, 40, 40), 2)
    sheet = cv2.imread(str(review / "sheet_001.png"))
    sheet[y1:y2, x1:x2] = hatch
    cv2.imwrite(str(review / "sheet_001.png"), sheet)
    new = cv2.imread(str(remove_stair(review, "A", 1) / "sheet_001.png"))
    inside = (max(y1, 820) + 3, min(y2, 1080) - 3, max(x1, 1220) + 3, min(x2, 1480) - 3)
    assert (sheet[inside[0]:inside[1], inside[2]:inside[3]] == new[inside[0]:inside[1], inside[2]:inside[3]]).mean() > 0.9
    assert (cv2.cvtColor(new[830:1020, 1230:1350], cv2.COLOR_BGR2GRAY) < 128).mean() < 0.002, "treads beside it still go"


def test_a_grey_well_and_its_tread_stubs_go_though_the_outline_stops_at_the_well(well):
    copy = remove_stair(well, "A", 1)
    old, new = (cv2.imread(str(p / "sheet_001.png")) for p in (well, copy))
    assert (cv2.cvtColor(new[825:1078, 1222:1478], cv2.COLOR_BGR2GRAY) < 225).mean() < 0.01
    assert (old[784:796, 1190:1510] == new[784:796, 1190:1510]).all(), "the wall round it stays"
    run(copy)
    assert any(f["check"] == 9 and f["sheet"] == "1" for f in read_json(copy / "review.json")["findings"])


def test_a_stair_the_removal_cannot_clear_is_refused_and_leaves_nothing(walled):
    before = folders(walled)
    with pytest.raises(ValueError, match="could not be removed cleanly"):
        remove_stair(walled, "A", 1)
    assert folders(walled) == before


def test_the_ink_test_is_the_stair_rules_own_cut(review, monkeypatch):
    monkeypatch.setattr(testtool, "ink_ratio", lambda *a, **k: ({1: INK_LOW}, {}))
    with pytest.raises(ValueError):
        remove_stair(review, "A", 1)
    monkeypatch.setattr(testtool, "ink_ratio", lambda *a, **k: ({1: INK_LOW - 0.01}, {}))
    assert remove_stair(review, "A", 1).is_dir()


def test_a_box_at_the_sheet_edge_stays_on_the_sheet(review):
    verticals = read_json(review / "stage_verticals.json")
    verticals["verticals"][0]["floors"]["1"] |= {"point": [100.0, 100.0], "box": [40.0, 60.0, 260.0, 240.0], "mask": None}
    write_json(review / "stage_verticals.json", verticals)
    sheet = cv2.imread(str(review / "sheet_001.png"))
    cv2.rectangle(sheet, (70, 70), (130, 130), (20, 20, 20), 2)
    cv2.imwrite(str(review / "sheet_001.png"), sheet)
    new = cv2.imread(str(remove_stair(review, "A", 1) / "sheet_001.png"))
    assert (cv2.cvtColor(new[60:140, 60:140], cv2.COLOR_BGR2GRAY) < 128).mean() < 0.002
    assert (new[:, 400:700] == sheet[:, 400:700]).all(), "a box past the sheet's edge must not wrap to the far side"


def test_only_the_named_sheet_is_edited(review):
    copy = remove_stair(review, "A", 1)
    assert (copy / "sheet_000.png").read_bytes() == (review / "sheet_000.png").read_bytes()
    assert (copy / "sheet_002.png").read_bytes() == (review / "sheet_002.png").read_bytes()
    assert (copy / "sheet_001.png").read_bytes() != (review / "sheet_001.png").read_bytes()


def test_a_cell_with_no_outline_uses_the_box_findings_draw_round_its_place(review):
    verticals = read_json(review / "stage_verticals.json")
    verticals["verticals"][0]["floors"]["1"] |= {"box": None, "mask": None}
    write_json(review / "stage_verticals.json", verticals)
    copy = remove_stair(review, "A", 1)
    treads = cv2.imread(str(copy / "sheet_001.png"))[860:1040, 1250:1450]
    assert (cv2.cvtColor(treads, cv2.COLOR_BGR2GRAY) < 128).mean() < 0.002


def test_removal_produces_a_gap_finding(review):
    assert not any(f["check"] == 9 for f in read_json(review / "review.json")["findings"])
    copy = remove_stair(review, "A", 1)
    run(copy)
    r = read_json(copy / "review.json")
    assert r["verticals"][0]["floors"]["1"]["state"] in ("check", "missing")
    assert [row["floors"]["1"]["state"] for row in r["verticals"]] != ["found"]
    gap = next(f for f in r["findings"] if f["check"] == 9)
    assert gap["sheet"] == "1" and gap["vertical"] == "A"
    assert gap["crop"].endswith(f"crop_{gap['comment_id']}.jpg") and (copy / f"crop_{gap['comment_id']}.jpg").is_file()
    assert read_json(copy / "stage_verticals.json")["verticals"][0]["floors"]["0"]["state"] == "found"


def test_the_screen_gets_the_edited_sheet(review):
    copy = remove_stair(review, "A", 1)
    run(copy)
    old, new = (cv2.imread(str(p / "sheet_001.jpg")) for p in (review, copy))
    assert ink(old[533:733, 813:987]).mean() > 3 * ink(new[533:733, 813:987]).mean()


def test_a_removal_at_the_discharge_level_raises_the_discharge_finding(review):
    copy = remove_stair(review, "A", 0)
    run(copy)
    r = read_json(copy / "review.json")
    assert r["verticals"][0]["floors"]["0"]["state"] != "found"
    assert any(f["check"] == 10 and f["sheet"] == "0" for f in r["findings"])


def test_it_is_labelled_a_test_on_the_review_the_list_and_the_letter(client, review):
    copy = remove_stair(review, "A", 1)
    run(copy)
    r = read_json(copy / "review.json")
    assert "Test" in r["project"]["note"] and "SECOND FLOOR" in r["project"]["note"]
    assert r["project"]["name"].startswith("Test copy") and "Test" in r["letter"]["note"]
    assert r["project"]["cycle"] == 1 and r["parent"] is None
    assert next(row["name"] for row in client.get("/api/reviews").json() if row["id"] == copy.name).startswith("Test copy")


def test_the_note_cannot_name_another_stair_when_the_rows_change_order(review):
    """Row letters are assigned again on each run, so a letter in the note can point at a stair that reads Found."""
    copy = remove_stair(review, "A", 1)

    def another_stair_first(kind, **request):
        if request["messages"][-1]["content"][0]["text"].startswith(FLOOR_P[:40]):
            other = {"name": "Side stair", "floors": [{"floor": f"F{k + 1}", "present": True, "x": 150, "y": 150} for k in range(3)]}
            return {"text": json.dumps({"stairs": [other, *json.loads(compare_reply())["stairs"]], "shafts": []}), "input": 1000, "output": 500}
        return fake(kind, **request)

    meta.set_transport(another_stair_first)
    run(copy)
    r = read_json(copy / "review.json")
    gap = next(f for f in r["findings"] if f["check"] == 9)
    assert gap["vertical"] == "B", "the removed stair, row A in the original, has a new letter here"
    assert "SECOND FLOOR" in gap["title"] and "SECOND FLOOR" in r["project"]["note"]
    assert not re.search(r"Stair [A-Z]\b", r["project"]["note"]) and not re.search(r"Stair [A-Z]\b", r["letter"]["note"])


def test_a_copy_of_a_second_review_keeps_its_cycle(review):
    child = build(review.parent / ("c" * 32))
    write_json(child / "stage_upload.json", {"name": "Second set", "parent": review.name})
    run(child)
    assert read_json(child / "review.json")["project"]["cycle"] == 2
    copy = remove_stair(child, "A", 1)
    run(copy)
    assert read_json(copy / "review.json")["project"]["cycle"] == 2


def test_a_second_review_of_a_test_review_is_labelled_too(review):
    copy = remove_stair(review, "A", 1)
    run(copy)
    child = build(review.parent / ("c" * 32))
    write_json(child / "stage_upload.json", {"name": "Second set", "parent": copy.name})
    run(child)
    assert "Test" in read_json(child / "review.json")["project"]["note"]


def test_comment_ids_and_the_examiners_choices_carry_to_the_copy(review):
    assemble(review, "seattle", "no", 1)
    before = {f["key"]: f["id"] for f in read_json(review / "review.json")["findings"]}
    copy = remove_stair(review, "A", 1)
    run(copy)
    r = read_json(copy / "review.json")
    assert (r["project"]["jurisdiction"], r["project"]["sprinklers"], r["project"]["discharge_level"]) == ("seattle", "no", "1")
    assert all(before[f["key"]] == f["id"] for f in r["findings"] if f["key"] in before)
    new = [f["id"] for f in r["findings"] if f["key"] not in before]
    assert new and all(int(i.rsplit("-", 1)[1]) > len(before) for i in new)


def test_the_run_goes_through_the_normal_meter(review):
    copy = remove_stair(review, "A", 1)
    run(copy)
    assert read_json(copy / "status.json")["usd"] > 0
    assert sum(json.loads(line)["usd"] for line in (review.parent / "ledger.jsonl").read_text().splitlines()) > 0
    assert read_json(copy / "review.json")["cost"]["usd"] > 0


@pytest.mark.parametrize("vertical, sheet", [("Z", 1), ("A", 7), ("../A", 1), ("", 1)])
def test_a_stair_or_sheet_the_review_does_not_hold_is_refused_and_leaves_nothing(review, vertical, sheet):
    before = folders(review)
    with pytest.raises(ValueError):
        remove_stair(review, vertical, sheet)
    assert folders(review) == before


def test_a_floor_where_no_finding_can_appear_is_refused(review):
    with pytest.raises(ValueError):
        remove_stair(review, "A", 2)   # the top floor: the stair just reads as ending lower


def test_a_bottom_floor_that_is_not_the_discharge_level_is_refused(review):
    write_json(review / "stage_floors.json", read_json(review / "stage_floors.json") | {"discharge": 1})
    with pytest.raises(ValueError):
        remove_stair(review, "A", 0)   # the stair just reads as starting higher


def test_the_examiners_discharge_level_outranks_the_floor_label(review):
    assemble(review, "houston", None, 1)
    with pytest.raises(ValueError):
        remove_stair(review, "A", 0)   # the label says this floor, and the examiner chose the next


def test_a_discharge_level_with_no_stair_above_it_is_refused(review):
    write_json(review / "stage_floors.json", read_json(review / "stage_floors.json") | {"discharge": 2})
    with pytest.raises(ValueError):
        remove_stair(review, "A", 2)


def test_the_examiners_keep_and_drop_carry_to_the_copy(review):
    write_json(review / "decisions.json", {"a-key": "drop"})
    assert read_json(remove_stair(review, "A", 1) / "decisions.json") == {"a-key": "drop"}


def test_a_cell_that_is_not_found_is_refused(review):
    verticals = read_json(review / "stage_verticals.json")
    verticals["verticals"][0]["floors"]["1"]["state"] = "check"
    write_json(review / "stage_verticals.json", verticals)
    with pytest.raises(ValueError):
        remove_stair(review, "A", 1)


def test_the_route_runs_the_copy_and_returns_its_id(client, review):
    r = client.post(f"/api/reviews/{RID}/remove", json={"vertical": "A", "sheet": 1})
    assert r.status_code == 200
    rid = r.json()["id"]
    assert rid != RID and finished(client, rid)["state"] == "done"
    assert any(f["check"] == 9 for f in read_json(review.parent / rid / "review.json")["findings"])


@pytest.mark.parametrize("body", [{"vertical": "Z", "sheet": 1}, {"vertical": "A", "sheet": 7}, {"vertical": "A", "sheet": 2}, {"vertical": "../../secret", "sheet": 1}, {"vertical": "A", "sheet": "../0"}, {"vertical": "A"}, {"sheet": 1}, {}])
def test_the_route_refuses_a_bad_value_and_keeps_its_slot(client, review, monkeypatch, body):
    monkeypatch.setattr(server, "waiting", threading.BoundedSemaphore(1))
    before = folders(review)
    assert client.post(f"/api/reviews/{RID}/remove", json=body).status_code == 422
    assert folders(review) == before
    assert server.waiting.acquire(blocking=False)


def test_the_route_refuses_a_stair_it_cannot_clear_before_any_spend(client, walled, monkeypatch):
    monkeypatch.setattr(server, "waiting", threading.BoundedSemaphore(1))
    meta.set_transport(lambda *a, **k: pytest.fail("a refused removal calls no model"))
    before = folders(walled)
    r = client.post(f"/api/reviews/{RID}/remove", json={"vertical": "A", "sheet": 1})
    assert r.status_code == 422 and "could not be removed cleanly" in r.json()["detail"]
    assert folders(walled) == before
    assert server.waiting.acquire(blocking=False)


def test_the_route_holds_the_edit_lock_while_it_copies(client, monkeypatch):
    held, real = [], server.remove_stair

    def spy(*args):
        held.append(server.edits.locked())
        return real(*args)

    monkeypatch.setattr(server, "remove_stair", spy)
    finished(client, client.post(f"/api/reviews/{RID}/remove", json={"vertical": "A", "sheet": 1}).json()["id"])
    assert held == [True]


def test_the_route_refuses_a_review_still_running(client, review):
    write_json(review / "status.json", {"state": "running"})
    assert client.post(f"/api/reviews/{RID}/remove", json={"vertical": "A", "sheet": 1}).status_code == 409
    assert folders(review) == [RID]


def test_the_route_refuses_a_bad_id(client):
    assert client.post("/api/reviews/not-an-id/remove", json={"vertical": "A", "sheet": 1}).status_code == 404


def test_the_route_needs_the_key(review, monkeypatch):
    monkeypatch.setattr(server, "DATA", review.parent)
    monkeypatch.setenv("STOREY_ACCESS_KEY", "test-key")
    assert TestClient(server.app, base_url="http://127.0.0.1").post(f"/api/reviews/{RID}/remove", json={"vertical": "A", "sheet": 1}).status_code == 401


@pytest.mark.parametrize("headers", [{"origin": "https://evil.example"}, {"sec-fetch-site": "cross-site"}])
def test_the_route_refuses_other_sites(client, review, headers):
    assert client.post(f"/api/reviews/{RID}/remove", headers=headers, json={"vertical": "A", "sheet": 1}).status_code == 403
    assert folders(review) == [RID]


def test_a_busy_server_turns_the_tool_away(client, review, monkeypatch):
    monkeypatch.setattr(server, "waiting", threading.BoundedSemaphore(1))
    server.waiting.acquire()
    assert client.post(f"/api/reviews/{RID}/remove", json={"vertical": "A", "sheet": 1}).status_code == 429
    assert folders(review) == [RID]


def test_the_tools_run_gives_its_slot_back(client, monkeypatch):
    monkeypatch.setattr(server, "waiting", threading.BoundedSemaphore(1))
    rid = client.post(f"/api/reviews/{RID}/remove", json={"vertical": "A", "sheet": 1}).json()["id"]
    finished(client, rid)
    assert server.waiting.acquire(blocking=False)


