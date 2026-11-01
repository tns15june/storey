"""The second review, on real ingest renders of a vector PDF and the real registration, with Spark stubbed. The two sets differ only where a test says so."""
import json
import random
from types import SimpleNamespace

import cv2
import numpy as np
import pymupdf
import pytest

from storey import findings, floors, ingest, meta, pipeline, recheck
from storey.assemble import CROP_PAD, assemble, evaluate_stage
from storey.meta import Meter
from storey.pipeline import Context, read_json, run, write_json
from storey.recheck import CLOUDS_DRAFT, NOT_COMPARED, recheck_stage
from storey.verticals import verticals_stage

PARENT, CHILD = "a" * 32, "b" * 32
PAGE = (1296, 864)           # points, so a sheet of 3600 by 2400 px at ingest's 200 dpi
K = 200 / 72                 # sheet px per point
STAIR = (300, 200, 420, 280)     # points: a stair-like symbol, drawn only on the sets that say so
NOTE = (800, 500, 960, 560)      # points: a note, drawn only on the sets that say so
QUIET = (120, 560, 240, 640)     # points: a place nothing changes
TAG = (1180, 406, 1188, 420)     # points: one digit in the blank margin, 7 by 13 px on the 2000 px copy, the size of the comment on stories on the planned set
EMPTY = {"text": json.dumps({"clouds": [], "deltas": []}), "input": 100, "output": 50}


def px(rect):
    return [v * K for v in rect]


def plan(page, stair, note, digit=None, cloud=False):
    """Rooms that do not repeat, so registration has one answer."""
    rng = random.Random(7)
    w, h = page.rect.width, page.rect.height
    n = 0
    for x in range(40, int(w) - 200, 180):
        for y in range(40, int(h) - 140, 130):
            n += 1
            box = pymupdf.Rect(x + rng.randint(0, 24), y + rng.randint(0, 20), x + rng.randint(130, 170), y + rng.randint(90, 120))
            if box.intersects(pymupdf.Rect(STAIR)) or box.intersects(pymupdf.Rect(NOTE)):
                continue   # a place the sets differ in holds nothing else, so the ink there is all change
            page.draw_rect(box, color=(0, 0, 0), width=3)
            page.insert_text((box.x0 + 8, box.y0 + 24), f"ROOM {n}", fontsize=11)
            page.draw_line((box.x0 + 12, box.y1 - 16), (box.x1 - rng.randint(12, 40), box.y1 - 16), color=(0, 0, 0), width=1)
    if stair:
        for x in range(STAIR[0], STAIR[2], 10):
            page.draw_line((x, STAIR[1]), (x, STAIR[3]), color=(0, 0, 0), width=1)
    if note:
        page.insert_text((NOTE[0] + 6, NOTE[1] + 24), "NEW PARTITION", fontsize=16)
        page.draw_line((NOTE[0] + 6, NOTE[1] + 40), (NOTE[2] - 8, NOTE[1] + 40), color=(0, 0, 0), width=2)
    if digit:
        page.insert_text((TAG[0], TAG[3]), digit, fontsize=12)
    if cloud:
        page.draw_rect(pymupdf.Rect(TAG) + (-50, -40, 50, 40), color=(0, 0, 0), width=2)   # a plain box, because only a cloud's ink counts here


def pdf(path, *pages, size=PAGE):
    doc = pymupdf.open()
    for sets in pages:
        plan(doc.new_page(width=size[0], height=size[1]), *sets)
    doc.save(path)
    return path


def read_rows(sheets, numbers):
    return {"sheets": [{"index": s.index, "number": number, "title": f"PLAN {number}", "cls": "floor_plan", "level": "", "scale": "", "plan_box": [0, 0, s.width, s.height], "read_ok": True}
                       for s, number in zip(sheets, numbers)]}


def finding(comment_id, sheet, box, key=None):
    return {"key": key or f"key-{comment_id}", "comment_id": comment_id, "check": 7, "severity": "fix", "discipline": "A", "sheet": sheet, "box": box, "title": f"Comment {comment_id}",
            "measured": "m", "required": "r", "rule": "", "evidence": "e", "vertical": None}


def folder(root, name, source, numbers, findings=(), parent=None):
    path = root / "reviews" / name
    path.mkdir(parents=True)
    sheets = ingest.ingest([source], path)
    write_json(path / "stage_read.json", read_rows(sheets, numbers))
    write_json(path / "stage_upload.json", {"name": "Set", "parent": parent})
    write_json(path / "status.json", {"state": "done"})
    write_json(path / "stage_floors.json", {"order": [s.index for s in sheets], "order_read": True, "reference": sheets[0].index, "transforms": {str(s.index): np.eye(3).tolist() for s in sheets}, "unaligned": [], "discharge": sheets[0].index, "frame_size": [960, 640], "stack": None, "not_checked": None})
    write_json(path / "stage_verticals.json", {"verticals": [{"id": "A", "name": "Stair", "established": True, "pos": None, "floors": {}}], "not_checked": []})
    write_json(path / "stage_declared.json", {"code": {}})
    write_json(path / "stage_vector.json", {"sheets": {}})
    write_json(path / "stage_evaluate.json", {"findings": list(findings), "open_items": [], "not_checked": [], "choices": {"jurisdiction": "houston", "sprinklers": None, "discharge": None},
                                              "ids": {f["key"]: f["comment_id"] for f in findings}})
    return path


def context(path):
    return Context(path, Meter(ledger=path.parent / "ledger.jsonl"), read_json(path / "sheets.json"), False)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    calls = []

    def fake(kind, **request):
        assert kind == "chat", "SAM takes no part in the second review"
        calls.append(request)
        return EMPTY

    monkeypatch.setattr(meta, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("a test reached the real client"))
    meta.set_transport(fake)
    yield calls
    meta.set_transport(None)


def pair(tmp_path, comments=(), raised=(), old=((True, False),), new=((False, False),), numbers=("A1.1",), new_numbers=None):
    """A parent review and its child, one PDF each. `raised` are the findings the child's checks raised again."""
    tmp_path.joinpath("in").mkdir(exist_ok=True)
    parent = folder(tmp_path, PARENT, pdf(tmp_path / "in/old.pdf", *old), numbers, comments)
    child = folder(tmp_path, CHILD, pdf(tmp_path / "in/new.pdf", *new), new_numbers or numbers, raised, parent=PARENT)
    return parent, child


def by_id(result):
    return {c["comment_id"]: c for c in result["comments"]}


def overlaps(box, rect):
    return box[0] < rect[2] and rect[0] < box[2] and box[1] < rect[3] and rect[1] < box[3]


def test_a_first_review_has_nothing_to_compare_and_calls_no_model(tmp_path, isolated):
    path = folder(tmp_path, PARENT, pdf(tmp_path / "set.pdf", (True, False)), ("A1.1",))
    assert recheck_stage(context(path)) == {}
    assert isolated == []


def test_changed_location_looks_resolved(tmp_path):
    parent, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    row = by_id(recheck_stage(context(child)))["A-01"]
    assert row["state"] == "resolved_looks" and row["why"]
    assert (child / row["old_crop"]).is_file() and (child / row["new_crop"]).is_file()
    assert row["old_crop"] != row["new_crop"]


def test_unchanged_location_stays_open(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(QUIET))])
    row = by_id(recheck_stage(context(child)))["A-01"]
    assert row["state"] == "open" and (child / row["new_crop"]).is_file()


def test_a_comment_the_checks_raise_again_stays_open_even_where_the_place_changed(tmp_path):
    comment = finding("A-01", 0, px(STAIR))
    _, child = pair(tmp_path, [comment], raised=[comment])
    row = by_id(recheck_stage(context(child)))["A-01"]
    assert row["state"] == "open" and "again" in row["why"]


def test_a_comment_with_no_place_never_reads_resolved(tmp_path):
    raised, gone = finding("A-01", None, None), finding("A-02", None, None)
    sheet_only = finding("A-03", 0, None)
    _, child = pair(tmp_path, [raised, gone, sheet_only], raised=[raised])
    rows = by_id(recheck_stage(context(child)))
    assert rows["A-01"]["state"] == "open"
    assert rows["A-02"]["state"] == "unknown" and rows["A-03"]["state"] == "unknown"
    assert rows["A-01"]["old_crop"] is None and rows["A-01"]["new_crop"] is None


def test_a_comment_the_examiner_dropped_is_not_compared(tmp_path):
    kept, dropped = finding("A-01", 0, px(QUIET)), finding("A-02", 0, px(STAIR))
    parent, child = pair(tmp_path, [kept, dropped])
    write_json(parent / "review.json", {"decisions": {dropped["key"]: "drop"}})
    assert list(by_id(recheck_stage(context(child)))) == ["A-01"]


def test_unmatched_sheet_is_listed_and_not_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(recheck, "register", lambda *a: pytest.fail("a sheet with no match has nothing to register"))
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))], new_numbers=("B9.9",))
    result = recheck_stage(context(child))
    assert set(result["unmatched_sheets"]) == {"A1.1", "B9.9"}
    assert by_id(result)["A-01"]["state"] == "unknown"
    assert result["changes"] == []


def test_a_number_two_sheets_share_matches_nothing(tmp_path):
    two = ((True, False), (True, True))
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))], old=two, new=two, numbers=("A1.1", "A1.1"))
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "unknown" and result["unmatched_sheets"] == ["A1.1"]


def test_a_sheet_that_does_not_register_reads_cannot_tell(tmp_path, monkeypatch, isolated):
    monkeypatch.setattr(recheck, "register", lambda *a: (np.eye(3), floors.GATE - 1))
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "unknown"
    assert result["unmatched_sheets"] == ["A1.1"] and result["changes"] == []
    assert len(isolated) == 12, "a sheet that cannot be compared is still read for clouds"


def test_a_fit_at_the_wrong_scale_reads_cannot_tell(tmp_path, monkeypatch):
    monkeypatch.setattr(recheck, "register", lambda *a: (np.diag([0.8, 0.8, 1.0]), 900))
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    assert by_id(recheck_stage(context(child)))["A-01"]["state"] == "unknown"


def test_a_wrong_fit_that_clears_the_gates_never_reads_resolved(tmp_path, monkeypatch):
    """Every comment on a wrong fit would read changed, so a sheet that differs nearly everywhere reads Cannot tell."""
    def wrong(old, new, *boxes):
        return np.array([[1, 0, 70.0], [0, 1, 45.0], [0, 0, 1]]), 900

    monkeypatch.setattr(recheck, "register", wrong)
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "unknown" and result["changes"] == []


def test_running_out_of_time_reads_cannot_tell(tmp_path, monkeypatch):
    monkeypatch.setattr(recheck, "ALIGN_SECONDS", -1)
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "unknown" and result["unmatched_sheets"] == ["A1.1"]


def test_change_outside_comments_is_new_change(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(QUIET))], old=((False, False),), new=((False, True),))
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "open"
    assert [c["state"] for c in result["changes"]] == ["new_change"]
    change = result["changes"][0]
    assert change["sheet"] == 0 and change["note"] and overlaps(change["box"], px(NOTE))


def test_a_change_a_comment_accounts_for_is_not_listed(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    assert recheck_stage(context(child))["changes"] == []


def test_one_stair_is_one_change(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(QUIET))])
    result = recheck_stage(context(child))
    assert len(result["changes"]) == 1 and overlaps(result["changes"][0]["box"], px(STAIR))


def test_a_change_inside_a_box_too_loose_for_the_cut_is_still_a_changed_place(tmp_path, monkeypatch):
    """A stair comment's box is the fallback round its place, so the ink in it is mostly not the stair's."""
    monkeypatch.setattr(recheck, "_changed", lambda a, b, box: False)
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "resolved_looks" and result["changes"] == []


def test_a_change_inside_a_loose_box_leaves_open_a_comment_the_checks_raise_again(tmp_path, monkeypatch):
    monkeypatch.setattr(recheck, "_changed", lambda a, b, box: False)
    comment = finding("A-01", 0, px(STAIR))
    _, child = pair(tmp_path, [comment], raised=[comment])
    assert by_id(recheck_stage(context(child)))["A-01"]["state"] == "open"


def test_a_change_far_larger_than_a_comment_is_listed_and_names_the_comment(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px((STAIR[0], STAIR[1], STAIR[0] + 30, STAIR[1] + 30)))])
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "resolved_looks"
    assert [c["state"] for c in result["changes"]] == ["new_change"]
    assert "A-01" in result["changes"][0]["note"] and "no comment covers" not in result["changes"][0]["note"]


def test_empty_cloud_list_is_not_no_change(tmp_path, isolated):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))], old=((True, False),), new=((False, True),))
    result = recheck_stage(context(child))
    assert isolated, "the sheet changed, so its tiles are read"
    assert result["clouds"] == [] and result["changes"]
    assert by_id(result)["A-01"]["state"] == "resolved_looks"
    assert CLOUDS_DRAFT in result["not_checked"]


def test_a_cloud_never_decides_a_state(tmp_path, isolated):
    """A cloud over every tile, on a sheet where one place changed and one did not."""
    meta.set_transport(lambda kind, **request: {"text": json.dumps({"clouds": [{"x1": 0, "y1": 0, "x2": 1000, "y2": 1000, "delta": "2"}], "deltas": []})})
    _, child = pair(tmp_path, [finding("A-01", 0, px(QUIET)), finding("A-02", 0, px(STAIR))], old=((True, False),), new=((False, True),))
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "open" and by_id(result)["A-02"]["state"] == "resolved_looks"
    assert result["clouds"] and [c["state"] for c in result["changes"]] == ["new_change"]
    assert "cloud" in result["changes"][0]["note"]


def test_two_renders_of_one_pdf_at_real_sizes_show_no_change(tmp_path, monkeypatch, isolated):
    """A sheet of 6000 px, the most ingest writes, against a render of the same PDF at 5900 px."""
    tmp_path.joinpath("in").mkdir()
    source = pdf(tmp_path / "in/set.pdf", (True, True), size=(2592, 1728))
    parent = folder(tmp_path, PARENT, source, ("A1.1",), [finding("A-01", 0, [v * 6000 / 2592 for v in STAIR])])
    monkeypatch.setattr(ingest, "MAX_EDGE", 5900)
    child = folder(tmp_path, CHILD, source, ("A1.1",), parent=PARENT)
    assert read_json(parent / "sheets.json")[0]["width"] == 6000 and read_json(child / "sheets.json")[0]["width"] == 5900
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "open"
    assert result["changes"] == [] and result["unmatched_sheets"] == [] and isolated == []


def test_a_real_size_change_is_found_on_the_new_sheets_pixels(tmp_path, monkeypatch):
    tmp_path.joinpath("in").mkdir()
    scale = 6000 / 2592
    parent = folder(tmp_path, PARENT, pdf(tmp_path / "in/a.pdf", (True, False), size=(2592, 1728)), ("A1.1",), [finding("A-01", 0, [v * scale for v in STAIR])])
    monkeypatch.setattr(ingest, "MAX_EDGE", 5400)
    child = folder(tmp_path, CHILD, pdf(tmp_path / "in/b.pdf", (False, True), size=(2592, 1728)), ("A1.1",), parent=PARENT)
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "resolved_looks"
    assert any(overlaps(c["box"], [v * 5400 / 2592 for v in NOTE]) for c in result["changes"])


def test_an_identical_sheet_is_neither_registered_nor_read_for_clouds(tmp_path, monkeypatch, isolated):
    monkeypatch.setattr(recheck, "register", lambda *a: pytest.fail("identical sheets need no registration"))
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))], new=((True, False),))
    result = recheck_stage(context(child))
    row = by_id(result)["A-01"]
    assert row["state"] == "open" and (child / row["old_crop"]).is_file()
    assert result["changes"] == [] and isolated == []


def test_a_new_sheet_with_no_match_is_read_for_clouds(tmp_path, isolated):
    _, child = pair(tmp_path, new_numbers=("A1.2",))
    recheck_stage(context(child))
    assert len(isolated) == 12


def tile_replies(*clouds):
    """Each cloud goes to the tile its grid row names, by position in the tile list."""
    seen = iter(range(100))

    def fake(kind, **request):
        n = next(seen)
        return {"text": json.dumps({"clouds": clouds[n] if n < len(clouds) else [], "deltas": []}), "input": 10, "output": 10}
    return fake


def test_tiles_cover_the_sheet_with_overlap_and_turn_with_it():
    wide = recheck._tiles(4000, 3000)
    assert len(wide) == 12 and wide[0][:2] == (0, 0)
    assert max(t[2] for t in wide) == pytest.approx(4000) and max(t[3] for t in wide) == pytest.approx(3000)
    assert wide[1][1] < wide[0][3]
    assert sorted({t[0] for t in recheck._tiles(3000, 4000)}) == sorted({t[1] for t in wide})


def test_cloud_boxes_map_from_the_tile_grid_to_sheet_pixels():
    tile = (1000.0, 500.0, 2000.0, 1500.0)
    assert recheck._clouds_in({"clouds": [{"x1": 100, "y1": 200, "x2": 300, "y2": 400, "delta": " 2 "}]}, tile) == [{"box": [1100.0, 700.0, 1300.0, 900.0], "delta": "2"}]


@pytest.mark.parametrize("cloud", [{"x1": 300, "y1": 200, "x2": 100, "y2": 400, "delta": ""}, {"x1": 100, "y1": 200, "x2": 100, "y2": 400, "delta": ""}])
def test_a_box_that_encloses_nothing_is_left_out(cloud):
    assert recheck._clouds_in({"clouds": [cloud]}, (0.0, 0.0, 1000.0, 1000.0)) == []


def test_an_empty_delta_is_none_and_a_long_one_is_cut():
    rows = recheck._clouds_in({"clouds": [{"x1": 0, "y1": 0, "x2": 5, "y2": 5, "delta": ""}, {"x1": 0, "y1": 0, "x2": 5, "y2": 5, "delta": "9" * 500}]}, (0.0, 0.0, 1000.0, 1000.0))
    assert rows[0]["delta"] is None and len(rows[1]["delta"]) == recheck.DELTA_MAX


def test_repeats_across_a_tile_overlap_count_once():
    assert len(recheck._merged([{"box": [100, 100, 200, 200], "delta": "1"}, {"box": [102, 100, 200, 198], "delta": "1"}])) == 1


def test_boxes_that_touch_across_a_tile_edge_become_one():
    assert recheck._merged([{"box": [100, 100, 200, 200], "delta": None}, {"box": [200, 120, 300, 180], "delta": "3"}, {"box": [900, 900, 950, 950], "delta": "4"}]) == [
        {"box": [100, 100, 300, 200], "delta": "3"}, {"box": [900, 900, 950, 950], "delta": "4"}]


def test_a_chain_of_touching_boxes_becomes_one():
    assert len(recheck._merged([{"box": [0, 0, 10, 10], "delta": None}, {"box": [10, 0, 20, 10], "delta": None}, {"box": [20, 0, 30, 10], "delta": None}])) == 1


def test_a_malformed_tile_reply_costs_that_tile_only(tmp_path):
    def fake(kind, **request):
        return {"text": "{\"clouds\": [{\"x1\": \"left\"}], \"deltas\": []}", "input": 10, "output": 10}

    meta.set_transport(fake)
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))], old=((True, False),), new=((False, True),))
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "resolved_looks" and result["clouds"] == []
    assert any("12 of 12" in line for line in result["not_checked"])


def test_an_empty_tile_reply_is_counted_as_unread(tmp_path):
    meta.set_transport(lambda kind, **request: {"text": None})
    _, child = pair(tmp_path, old=((True, False),), new=((False, True),))
    assert any("12 of 12" in line for line in recheck_stage(context(child))["not_checked"])


def test_a_budget_stop_keeps_every_comment_state_and_says_so(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))], old=((True, False),), new=((False, True),))
    ctx = context(child)
    ctx.meter = Meter(cap_usd=0, ledger=ctx.meter.ledger)
    result = recheck_stage(ctx)
    assert by_id(result)["A-01"]["state"] == "resolved_looks" and result["changes"]
    assert any("spend cap" in line for line in result["not_checked"])


def test_comment_ids_persist(tmp_path):
    """A comment raised again takes its parent's id, a new one takes the next, and a comment no longer raised keeps its row."""
    kept, gone = finding("A-07", 0, px(QUIET)), finding("A-08", 0, px(STAIR))
    _, child = pair(tmp_path, [kept, gone], raised=[kept, finding("A-09", 0, None)])
    write_json(child / "stage_recheck.json", recheck_stage(context(child)))
    r = assemble(child)
    assert [c["comment_id"] for c in r["recheck"]["comments"]] == ["A-07", "A-08"]
    assert {f["id"]: f["state"] for f in r["findings"]} == {"A-07": "open", "A-09": None}
    assert {c["comment_id"]: c["state"] for c in r["recheck"]["comments"]} == {"A-07": "open", "A-08": "resolved_looks"}


def test_a_comment_row_carries_its_title_for_a_comment_no_longer_in_the_findings(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    assert by_id(recheck_stage(context(child)))["A-01"]["title"] == "Comment A-01"


def test_review_json_holds_the_recheck_block_in_each_sheets_jpeg_pixels(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))], old=((True, False),), new=((False, True),))
    result = recheck_stage(context(child))
    write_json(child / "stage_recheck.json", result | {"clouds": [{"sheet": 0, "box": [1800.0, 1200.0, 3600.0, 2400.0], "delta": None}]})
    block = assemble(child)["recheck"]
    assert block["clouds"] == [{"sheet": "0", "box": [1000.0, round(1200 * 1333 / 2400, 1), 2000.0, 1333.0], "delta": None}]   # 2400 px scale to a whole 1333
    assert block["changes"][0]["sheet"] == "0" and max(block["changes"][0]["box"]) <= 2000
    assert block["comments"][0]["old_crop"] == f"/api/reviews/{CHILD}/files/{result['comments'][0]['old_crop']}"
    assert block["unmatched_sheets"] == []


def test_a_crop_url_is_one_the_file_route_serves(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from storey import server
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    write_json(child / "stage_recheck.json", recheck_stage(context(child)))
    url = assemble(child)["recheck"]["comments"][0]["new_crop"]
    monkeypatch.setattr(server, "DATA", child.parent)
    monkeypatch.setenv("STOREY_ACCESS_KEY", "test-key")
    assert TestClient(server.app, base_url="http://127.0.0.1", headers={"x-storey-key": "test-key"}).get(url).status_code == 200


def test_a_redrawn_sheet_is_one_change_with_its_count(tmp_path, monkeypatch):
    monkeypatch.setattr(recheck, "MAX_CHANGES", 0)
    _, child = pair(tmp_path, [finding("A-01", 0, px(QUIET))])
    changes = recheck_stage(context(child))["changes"]
    assert len(changes) == 1 and "1 places" in changes[0]["note"]


def test_a_first_review_has_no_recheck_block(tmp_path):
    path = folder(tmp_path, PARENT, pdf(tmp_path / "set.pdf", (True, False)), ("A1.1",), [finding("A-01", 0, px(STAIR))])
    write_json(path / "stage_recheck.json", {})
    r = assemble(path)
    assert r["recheck"] is None and r["findings"][0]["state"] is None


def test_a_recheck_that_did_not_finish_says_so_and_sets_no_state(tmp_path):
    comment = finding("A-01", 0, px(STAIR))
    _, child = pair(tmp_path, [comment], raised=[comment])
    write_json(child / "stage_recheck.json", {"failed": "Compare with the last review did not finish."})
    r = assemble(child)
    assert r["recheck"] is None and r["findings"][0]["state"] is None
    assert any("recheck stage did not finish" in line for line in r["not_checked"])


def test_the_stage_runs_in_the_pipeline_and_a_failure_loses_no_review(tmp_path, monkeypatch):
    from storey import pipeline
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])

    def boom(ctx):
        raise RuntimeError("stopped")

    run(child, stages=[("recheck", "Compare", boom), pipeline.STAGES[-1]])
    assert "recheck stage did not finish" in " ".join(read_json(child / "review.json")["not_checked"])


def test_the_notes_reach_the_examiner(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    write_json(child / "stage_recheck.json", recheck_stage(context(child)))
    assert CLOUDS_DRAFT in assemble(child)["not_checked"]


def test_a_word_replaced_in_place_with_no_cloud_stays_still_open(tmp_path):
    """The known limit: a swap keeps most of the ink in the box, so with no cloud neither test finds a change here."""
    _, child = pair(tmp_path, [finding("A-01", 0, px(TAG))], old=((False, False, "6"),), new=((False, False, "5"),))
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "open" and result["changes"] == []


def test_a_word_replaced_in_place_inside_a_cloud_looks_resolved(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(TAG))], old=((False, False, "6"),), new=((False, False, "5", True),))
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "resolved_looks"
    assert len(result["changes"]) == 1 and "A-01" in result["changes"][0]["note"] and "no comment covers" not in result["changes"][0]["note"]


def test_a_change_beside_a_comment_and_clear_of_its_box_leaves_it_open(tmp_path):
    box = (NOTE[0] - 90, NOTE[1], NOTE[0] - 50, NOTE[1] + 40)
    _, child = pair(tmp_path, [finding("A-01", 0, px(box))], old=((False, False),), new=((False, True),))
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "open"
    assert [c["note"] for c in result["changes"]] == [recheck.NEW_CHANGE]


def stair_comment(check=9):
    return finding("A-01", 0, px(STAIR)) | {"check": check, "vertical": "A", "title": "Stair A is missing on SECOND FLOOR"}


def run_checks(child):
    """The child's evaluate stage over whatever stages it now holds, as the pipeline would run it."""
    write_json(child / "stage_evaluate.json", evaluate_stage(context(child)))


def state_after(child, comment="A-01"):
    row = by_id(recheck_stage(context(child)))[comment]
    return row["state"], row["why"]


def test_every_check_a_comment_can_carry_has_a_row_in_the_table():
    assert set(recheck.RAISED_BY) == set(findings.SEVERITY) - {12}
    assert {label for labels in recheck.RAISED_BY.values() for label in labels} == {label for label, _, _ in findings.CHECKS} - {"stair list", "reads that did not finish"}


def test_a_comment_with_no_table_row_never_reads_resolved(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR)) | {"check": 99}])
    assert state_after(child)[0] == "unknown"


def test_every_stair_reply_empty_is_cannot_tell_and_never_resolved(tmp_path):
    two = ((True, False), (True, False))
    _, child = pair(tmp_path, [stair_comment()], old=two, new=((False, False), (True, False)), numbers=("A1.1", "A1.2"))
    meta.set_transport(lambda kind, **request: {"text": None})
    write_json(child / "stage_verticals.json", verticals_stage(context(child)))
    assert read_json(child / "stage_verticals.json")["verticals"] == []
    run_checks(child)
    state, why = state_after(child)
    assert state == "unknown" and why == recheck.NOT_TOLD


def test_rows_that_no_two_runs_confirmed_prove_nothing(tmp_path):
    _, child = pair(tmp_path, [stair_comment()])
    write_json(child / "stage_verticals.json", {"verticals": [{"id": "A", "name": "Stair", "established": False, "pos": None, "floors": {}}], "not_checked": []})
    run_checks(child)
    assert state_after(child)[0] == "unknown"


@pytest.mark.parametrize("stage", ["verticals", "floors", "read"])
def test_a_stage_a_stair_check_reads_failed_is_cannot_tell(tmp_path, stage):
    _, child = pair(tmp_path, [stair_comment()])
    write_json(child / f"stage_{stage}.json", {"failed": "It did not finish."})
    run_checks(child)
    assert state_after(child)[0] == "unknown"


def test_the_evaluate_stage_failing_is_cannot_tell(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    write_json(child / "stage_evaluate.json", {"failed": "Run the checks did not finish."})
    assert state_after(child)[0] == "unknown"


def test_a_spend_cap_in_the_stair_stage_is_cannot_tell(tmp_path, monkeypatch):
    two = ((True, False), (True, False))
    _, child = pair(tmp_path, [stair_comment()], old=two, new=((False, False), (True, False)), numbers=("A1.1", "A1.2"))
    (child / "stage_verticals.json").unlink()
    (child / "stage_evaluate.json").unlink()
    monkeypatch.setattr(pipeline, "Meter", lambda ledger: Meter(cap_usd=0, ledger=ledger))
    run(child, stages=[stage for stage in pipeline.STAGES if stage[0] in ("verticals", "evaluate", "recheck")])
    assert "spend cap" in read_json(child / "stage_verticals.json")["failed"]
    assert by_id(read_json(child / "stage_recheck.json"))["A-01"]["state"] == "unknown"


def test_the_declared_stage_failing_on_a_door_comment_is_cannot_tell(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    write_json(child / "stage_declared.json", {"failed": "Read what the set declares did not finish."})
    run_checks(child)
    assert state_after(child)[0] == "unknown"


def test_a_stage_with_no_result_is_cannot_tell(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    (child / "stage_declared.json").unlink()
    run_checks(child)
    assert state_after(child)[0] == "unknown"


def test_a_comment_on_a_stage_that_failed_with_no_bearing_on_it_can_still_read_resolved(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR)) | {"check": 3}])
    write_json(child / "stage_verticals.json", {"failed": "Follow the stairs did not finish."})
    run_checks(child)
    assert state_after(child)[0] == "resolved_looks"


@pytest.mark.parametrize("floors_stage", [{"order": [], "order_read": True, "not_checked": "No floor name was read on A1.1, so stairs were not followed there."},
                                          {"order": [0], "order_read": True, "unaligned": [0], "not_checked": None}])
def test_a_sheet_outside_the_new_floor_order_or_not_aligned_is_cannot_tell(tmp_path, floors_stage):
    """Nothing failed anywhere, and the stair checks did not look at this floor."""
    _, child = pair(tmp_path, [stair_comment()])
    write_json(child / "stage_floors.json", floors_stage)
    run_checks(child)
    assert state_after(child)[0] == "unknown"


def test_a_comment_on_a_sheet_with_no_floor_can_read_resolved(tmp_path):
    """A code sheet is in no floor order, and a stories comment is settled there."""
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR)) | {"check": 5}])
    write_json(child / "stage_floors.json", {"order": [], "order_read": True, "not_checked": None})
    run_checks(child)
    assert state_after(child)[0] == "resolved_looks"


def test_the_floor_order_is_read_on_the_new_sets_sheet_numbers(tmp_path):
    """The comment is on sheet 1 of the old set, and that sheet is sheet 0 of the new one."""
    _, child = pair(tmp_path, [stair_comment() | {"sheet": 1}], old=((False, False), (True, False)), new=((False, False), (False, False)), numbers=("A1.0", "A1.1"), new_numbers=("A1.1", "A1.0"))
    write_json(child / "stage_floors.json", {"order": [0], "order_read": True, "unaligned": [], "discharge": 0, "not_checked": None})
    run_checks(child)
    assert state_after(child)[0] == "resolved_looks"


def test_a_check_that_crashed_is_cannot_tell(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR)) | {"check": 3}])
    write_json(child / "stage_evaluate.json", read_json(child / "stage_evaluate.json") | {"not_checked": ["The sheet titles check did not finish."]})
    assert state_after(child)[0] == "unknown"


def test_a_rule_check_a_dwelling_skips_is_cannot_tell(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR)) | {"check": 14}])
    write_json(child / "stage_evaluate.json", read_json(child / "stage_evaluate.json") | {"not_checked": [findings.DWELLING]})
    assert state_after(child)[0] == "unknown"


@pytest.mark.parametrize("sheet", [0, None])
def test_a_doubt_the_same_check_raised_here_is_cannot_tell(tmp_path, sheet):
    """A corridor that could not be measured leaves its comment unraised, under another key."""
    doubt = finding("A-02", sheet, None) | {"check": 14, "severity": "unknown"}
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR)) | {"check": 14}], raised=[doubt])
    assert state_after(child)[0] == "unknown"


def test_a_doubt_on_another_check_or_sheet_changes_nothing(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR)) | {"check": 14}], raised=[finding("A-02", 0, None) | {"check": 13, "severity": "unknown"}, finding("A-03", 3, None) | {"check": 14, "severity": "unknown"}])
    assert state_after(child)[0] == "resolved_looks"


def test_a_place_that_did_not_change_says_nothing_of_a_check_that_did_not_run(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(QUIET))])
    write_json(child / "stage_evaluate.json", {"failed": "Run the checks did not finish."})
    assert state_after(child) == ("open", recheck.STILL)


def test_a_parent_whose_checks_did_not_finish_compares_nothing_and_says_so(tmp_path):
    parent, child = pair(tmp_path)
    write_json(parent / "stage_evaluate.json", {"failed": "Run the checks did not finish."})
    result = recheck_stage(context(child))
    assert result["comments"] == [] and NOT_COMPARED in result["not_checked"]


def test_a_sheet_the_read_lost_matches_nothing(tmp_path):
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    rows = read_json(child / "stage_read.json")
    rows["sheets"][0]["read_ok"] = False
    write_json(child / "stage_read.json", rows)
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "unknown" and result["unmatched_sheets"] == ["A1.1"]


def test_registration_reads_the_whole_sheet_and_not_the_plan_box(tmp_path, monkeypatch):
    """Spark's plan box can cover 13% of a sheet's width, and a fit from there is extrapolated across the rest. Both sheets are the same sheet."""
    seen, real = [], recheck.register

    def spy(old, new, *boxes):
        seen.append(boxes)
        return real(old, new, *boxes)

    monkeypatch.setattr(recheck, "register", spy)
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))])
    rows = read_json(child / "stage_read.json")
    rows["sheets"][0]["plan_box"] = [100, 100, 500, 900]
    write_json(child / "stage_read.json", rows)
    recheck_stage(context(child))
    assert seen == [([0, 0, 3600, 2400], [0, 0, 3600, 2400])]


def rooms(shift=0):
    """A 2000 px copy with rooms and lines of text, moved by `shift` px."""
    im = np.full((1333, 2000, 3), 255, np.uint8)
    for x in range(100, 1800, 170):
        cv2.rectangle(im, (x + shift, 200 + shift), (x + 130 + shift, 1100 + shift), (0, 0, 0), 2)
        for y in range(240, 1060, 26):
            cv2.line(im, (x + 14 + shift, y + shift), (x + 112 + shift, y + shift), (0, 0, 0), 2)
    return im


def regions(old, new):
    return recheck._regions(recheck._soft_ink(old), recheck._soft_ink(new))


@pytest.mark.parametrize("shift", [1, 2])
def test_a_line_that_moved_a_pixel_or_two_is_not_a_change(shift):
    assert regions(rooms(), rooms(shift)) == []


def test_a_word_gone_from_a_busy_sheet_is_a_change():
    """A window round it holds other ink, so a ratio over the window reads no change. Capitol listed 6 of 36 deleted words that way."""
    new = rooms()
    new[500:512, 280:370] = 255
    found = regions(rooms(), new)
    assert len(found) == 1 and overlaps(found[0], (280, 500, 370, 512))


def test_a_frame_line_a_rescan_pushes_off_the_page_is_not_a_change(monkeypatch):
    """Sheets with a frame on the page edge gave up to 5 false regions per sheet through the lab's rescan without the erosion."""
    old = rooms()
    cv2.rectangle(old, (0, 0), (1999, 1332), (0, 0, 0), 1)
    turn = np.vstack([cv2.getRotationMatrix2D((1000, 666), 0.2, 1.0), [0, 0, 1]])
    new = cv2.warpPerspective(old, turn, (2000, 1333), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    monkeypatch.setattr(recheck, "register", lambda *a: (np.linalg.inv(turn), 900))
    _, a, b = recheck._fit(old, new, 1e18)
    assert recheck._regions(a, b) == []


def test_a_fit_at_a_scale_the_gate_refuses_is_not_compared_though_its_ink_agrees(monkeypatch):
    """A fit at 0.8 with new ink that matches it: only the scale gate can stop this one."""
    fit = np.diag([0.8, 0.8, 1.0])
    new = cv2.warpPerspective(rooms(), np.linalg.inv(fit), (2000, 1333), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    monkeypatch.setattr(recheck, "register", lambda *a: (fit, 900))
    assert recheck._fit(rooms(), new, 1e18) is None


def test_a_wrong_fit_on_a_sparse_sheet_is_not_compared(monkeypatch):
    """A sheet with so little ink that the per-box floor would skip the guard."""
    old = np.full((1333, 2000, 3), 250, np.uint8)
    for k, (x, y) in enumerate([(300, 300), (900, 500), (1500, 900), (600, 1000)]):
        cv2.putText(old, f"NOTE {k}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    assert 2 * recheck._soft_ink(old).mean() < recheck.FAINT
    monkeypatch.setattr(recheck, "register", lambda *a: (np.array([[1, 0, 70.0], [0, 1, 45.0], [0, 0, 1]]), 900))
    assert recheck._fit(old, old.copy(), 1e18) is None


def test_the_box_ratio_finds_a_faint_change_no_region_holds(tmp_path):
    """Grey under the stroke cut, gone from a comment's box: only the ratio finds it."""
    old, new = np.full((1333, 2000, 3), 255, np.uint8), np.full((1333, 2000, 3), 255, np.uint8)
    old[300:340, 300:400] = 225
    a, b = recheck._soft_ink(old), recheck._soft_ink(new)
    assert recheck._regions(a, b) == []
    rows, changes, _ = recheck._compare(SimpleNamespace(dir=tmp_path), [finding("A-01", 0, [300, 300, 400, 340])], set(), {"A-01"}, old, new, (np.eye(3), a, b), 0)
    assert rows[0]["state"] == "resolved_looks" and changes == []


def test_the_new_crop_is_the_new_sheet_at_the_mapped_place(tmp_path, monkeypatch):
    tmp_path.joinpath("in").mkdir()
    box = [v * 6000 / 2592 for v in STAIR]
    parent = folder(tmp_path, PARENT, pdf(tmp_path / "in/a.pdf", (True, False), size=(2592, 1728)), ("A1.1",), [finding("A-01", 0, box)])
    monkeypatch.setattr(ingest, "MAX_EDGE", 5400)
    child = folder(tmp_path, CHILD, pdf(tmp_path / "in/b.pdf", (False, False), size=(2592, 1728)), ("A1.1",), parent=PARENT)
    row = by_id(recheck_stage(context(child)))["A-01"]
    crop = cv2.imread(str(child / row["new_crop"]))
    sheet = cv2.imread(str(child / read_json(child / "sheets.json")[0]["image"]))
    x1, y1, x2, y2 = (int(v * 0.9) for v in box)
    want = sheet[y1 - CROP_PAD:y2 + CROP_PAD, x1 - CROP_PAD:x2 + CROP_PAD]
    assert abs(crop.shape[0] - want.shape[0]) <= 3 and abs(crop.shape[1] - want.shape[1]) <= 3
    assert np.abs(cv2.resize(crop, want.shape[1::-1]).astype(float) - want).mean() < 6


def test_an_infinite_coordinate_costs_that_tile_only(tmp_path):
    text = '{"clouds": [{"x1": 1e999, "y1": 0, "x2": 10, "y2": 10, "delta": ""}], "deltas": []}'
    meta.set_transport(lambda kind, **request: {"text": text, "input": 10, "output": 10})
    _, child = pair(tmp_path, [finding("A-01", 0, px(STAIR))], old=((True, False),), new=((False, True),))
    result = recheck_stage(context(child))
    assert by_id(result)["A-01"]["state"] == "resolved_looks" and any("12 of 12" in line for line in result["not_checked"])


def test_a_comment_back_in_the_findings_after_a_selector_change_reads_still_open(tmp_path):
    comment = finding("A-01", 0, px(STAIR))
    _, child = pair(tmp_path, [comment])
    write_json(child / "stage_recheck.json", recheck_stage(context(child)))
    assert assemble(child)["recheck"]["comments"][0]["state"] == "resolved_looks"
    write_json(child / "stage_evaluate.json", read_json(child / "stage_evaluate.json") | {"findings": [comment], "ids": {comment["key"]: "A-01"}})   # what a selector change stores
    r = assemble(child)
    assert [f["state"] for f in r["findings"]] == ["open"]
    row = r["recheck"]["comments"][0]
    assert row["state"] == "open" and "again" in row["why"]
