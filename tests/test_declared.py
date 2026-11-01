import base64
import collections
import json
import subprocess
import sys

import cv2
import httpx
import numpy as np
import openai
import pymupdf
import pytest

from storey import declared, meta
from storey.declared import (agree, code_fields, cover_index, declared_stage, drawing_labels, exit_sheet, floor_words, page_lines, plan_rooms, row_strips, schedules, select,
                             shape_tags, table_boxes, text_reads, text_tables, tiles, title_blocks, width_mm)
from storey.declared import room_same
from storey.meta import Meter
from storey.pipeline import Context, write_json

W, H = 2592, 1728
TB_START = "This image is one sheet of a building drawing set. Read its title block"
CODE_START = "This image is one sheet of a building drawing set. Read what"
PX = 2.0   # the fixture's sheet images are twice the page size
INDEX_START = "This image shows part of the sheet index"
DOOR_START = "This image contains a door schedule"
FINISH_START = "This image contains a room finish schedule"
OCC_START = "This image is an occupant load table"
LABEL_START = "This image is one sheet of a building drawing set. sheet_number: the drawing"
BAD = {"text": None}   # an empty reply, which spark refuses
FIELDS = {"code_edition": "2015 Michigan Residential Code", "occupancy": "R-3", "occupancy_mentions": [], "construction": "VB", "sprinklered": "no",
          "sprinkler_mentions": [], "stories": "2", "area": "1,728 SF", "occupant_load": "12"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(declared, "WORKERS", 1)   # a crop's first read then always comes before its second
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("a test reached the real client"))
    yield
    meta.set_transport(None)


def text(obj):
    return {"text": json.dumps(obj), "input": 100, "cached": 0, "output": 10, "finish": "stop"}


def prompt_of(request):
    content = request["messages"][0]["content"]
    return content if isinstance(content, str) else content[0]["text"]


def sheet_of(request):
    """The sheet a crop came from: every test sheet's bottom right corner is a grey that holds its index, and title crops sit there."""
    url = request["messages"][0]["content"][1]["image_url"]["url"]
    return int(cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)[-1, -1, 0])


def by_sheet(replies):
    return lambda request: replies[sheet_of(request)]


def by_crop(replies):
    """Replies by crop, numbered in the order first seen, so a test can lose one crop. An entry can be a list, one reply per read."""
    seen = {}

    def reply(request):
        return replies(seen.setdefault(request["messages"][0]["content"][1]["image_url"]["url"], len(seen)))
    return reply


class Model:
    """Replies by prompt. A reply can be a function of the request. A list gives its entries to the first, second and later calls with the same prompt and image, so read 1 and
    read 2 of one crop get their own replies however the threads run. An index with no answer in the test reads as empty."""

    def __init__(self, answers=None):
        self.answers, self.calls, self.seen = {INDEX_START: {"rows": []}} | (answers or {}), [], collections.Counter()

    def __call__(self, kind, **request):
        prompt = prompt_of(request)
        self.calls.append(prompt)
        same = json.dumps(request["messages"])
        nth = self.seen[same]
        self.seen[same] += 1
        for start, reply in self.answers.items():
            if prompt.startswith(start):
                reply = reply(request) if callable(reply) else reply
                if isinstance(reply, list):
                    reply = reply[min(nth, len(reply) - 1)]
                if isinstance(reply, Exception):
                    raise reply
                return text(reply) if "text" not in reply else {"input": 1, "cached": 0, "output": 1, "finish": "stop"} | reply
        raise AssertionError(f"no answer for: {prompt[:70]}")

    def count(self, prompt):
        return sum(p.startswith(prompt) for p in self.calls)


def line(t, x0, y0, size=12.6, along=True, width=None):
    return {"t": t, "x0": x0, "y0": y0, "x1": x0 + (width or len(t) * size * 0.6), "y1": y0 + size, "size": size, "dir": [1, 0] if along else [0, -1]}


def page(*lines, w=W, h=H, tags=()):
    return {"w": w, "h": h, "lines": list(lines), "tags": list(tags)}


def block(number, title, y=1560):
    """A title block in the right tenth of the sheet."""
    return [line(title, 2400, y), line(number, 2420, y + 60, 31.5)]


def table(heading, cols, rows, x=100, y=200, gap=15):
    """A heading, a header row and body rows as text lines."""
    out = [line(heading, x, y, 18.9)]
    step = 70
    out += [line(c, x + i * step, y + 40, 9.6, width=30) for i, c in enumerate(cols)]
    for n, row in enumerate(rows):
        out += [line(c, x + i * step, y + 40 + (n + 1) * gap, 9.6, width=30) for i, c in enumerate(row) if c]
    return out


DOORS = table("DOOR SCHEDULE", ["MARK", "SIZE", "SWING"], [["1", '36" x 80"', "LEFT"], ["2", '36" x 80"', "RIGHT"], ["3", '60" x 80"', "BIFOLD"]])


def sheet_record(i, w=W, h=H, kind="vector"):
    return {"index": i, "image": f"sheet_{i:03d}.png", "width": int(w * PX), "height": int(h * PX), "kind": kind, "text": "", "words": []}


def read_row(i, number="", title="", cls="other", level="", plan_box=None, read_ok=True):
    return {"index": i, "number": number, "title": title, "cls": cls, "level": level, "scale": "", "plan_box": plan_box or [0, 0, int(W * PX), int(H * PX)], "read_ok": read_ok}


def noise_sheet(i, h, w):
    """Noise, so no two tiles are the same image and collapse into one cached call, with a corner of grey i for sheet_of."""
    image = np.random.default_rng(i).integers(0, 255, (h, w, 3), np.uint8)
    image[int(0.8 * h):, int(0.7 * w):] = i
    return image


def review(tmp_path, rows, images=True, cap_usd=8.0, sizes=None):
    """A review folder with blank sheet images and a stored read stage."""
    sheets = []
    for row in rows:
        w, h = (sizes or {}).get(row["index"], (W, H))
        sheets.append(sheet_record(row["index"], w, h, "scan"))
        if images:
            cv2.imwrite(str(tmp_path / sheets[-1]["image"]), noise_sheet(row["index"], int(h * PX // 4), int(w * PX // 4)))
    write_json(tmp_path / "stage_read.json", {"sheets": rows})
    return Context(tmp_path, Meter(cap_usd=cap_usd, ledger=tmp_path / "ledger.jsonl"), sheets, False)


def write_pdf(path, draw, pages=1, rotation=0):
    with pymupdf.open() as doc:
        for i in range(pages):
            p = doc.new_page(width=W, height=H)
            draw(p, i)
            if rotation:
                p.set_rotation(rotation)
        doc.save(path)


def pdf_review(tmp_path, draw, rows, pages=1, rotation=0):
    (tmp_path / "upload").mkdir(exist_ok=True)
    write_pdf(tmp_path / "upload/0000.pdf", draw, pages, rotation)
    return review(tmp_path, rows)


# ---- the text layer ----

def test_non_breaking_space_in_title(tmp_path):
    def draw(p, i):
        p.insert_text((2400, 1560), "PLAN - LEVEL 1", fontsize=12.6)
        p.insert_text((2420, 1620), "A2.1", fontsize=31.5)
    write_pdf(tmp_path / "a.pdf", draw)
    with pymupdf.open(tmp_path / "a.pdf") as doc:
        lines = page_lines(doc[0])
    assert title_blocks({0: page(*lines)})[0]["title"] == "PLAN - LEVEL 1"


def test_rotated_page_lines_are_in_the_rendered_frame(tmp_path):
    write_pdf(tmp_path / "a.pdf", lambda p, i: p.insert_text((100, 300), "DOOR SCHEDULE", fontsize=18), rotation=90)
    with pymupdf.open(tmp_path / "a.pdf") as doc:
        got = page_lines(doc[0])[0]
        assert doc[0].rect.width == H
    # a 90 degree page puts the point (100, 300) at x = 1728 - 300 in the frame that was rendered
    assert abs(got["x0"] - (H - 300)) < 25 and abs(got["y0"] - 100) < 200
    assert got["dir"] == [0, 1]


def test_title_block_drops_lines_repeated_across_the_set():
    pages = {i: page(*block(f"A{i}.0", f"SHEET {'ABCDEFGHIJ'[i]}"), line("Smith Architects", 2400, 100)) for i in range(10)}
    got = title_blocks(pages)
    assert got[3]["number"] == "A3.0"
    assert got[3]["title"] == "SHEET D"


@pytest.mark.parametrize("first, second, want", [("{} FLOOR LIFE", "SAFETY PLAN", "{} FLOOR LIFE SAFETY PLAN"), ("FLOOR PLAN -", "{} FLOOR", "FLOOR PLAN - {} FLOOR")],
                         ids=["second-line", "first-line"])
def test_title_wrapped_onto_a_line_the_set_repeats_keeps_that_line(first, second, want):
    floors = ("FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH")
    pages = {n: page(line(first.format(f), 2400, 1500), line(second.format(f), 2400, 1515.6), line(f"G1.{n + 1}", 2420, 1620, 31.5), line("Smith Architects", 2400, 100))
             for n, f in enumerate(floors)}
    pages |= {n: page(*block(f"A{n}.0", f"SHEET {n}"), line("Smith Architects", 2400, 100)) for n in range(5, 14)}
    got = title_blocks(pages)
    assert [got[n]["title"] for n in range(6)] == [want.format(f) for f in floors] + ["SHEET 5"]


@pytest.mark.parametrize("extra", [None, line("PROJECT NO.", 2401, 1539.6, 14), line("SCALE", 2500, 1540), line("THE CAPITOL", 2410, 1467, along=False) | {"x1": 2422.6, "y1": 1550}],
                         ids=["as-printed", "other-size-just-above", "beside-not-over", "turned-just-above"])
def test_title_takes_a_repeated_line_only_from_its_own_paragraph(extra):
    """Capitol's strip as measured: the job number, on every sheet at the title's size, sits 18.8 pt above the title, and MAIN FLOOR wraps 3.2 pt onto PLAN, which three sheets print.
    A repeated line of another size, beside the title or turned on its side stays out too, though it ends within half a line of the title."""
    def strip(n, floor):
        own = [line(f"{floor} FLOOR", 2400.4, 1554.6) | {"y1": 1566.4}, line("PLAN", 2400.4, 1569.6)] if floor else [line(f"SHEET {n}", 2400.4, 1554.6)]
        return page(line("2023xx", 2401, 1524) | {"y1": 1535.8}, *own, line(f"A{n}.1", 2420.6, 1625.6, 37.7), *[x for x in [extra] if x])
    got = title_blocks({n: strip(n, floor) for n, floor in enumerate(["MAIN", "UPPER", "LOWER", *[None] * 7])})
    assert [got[n]["title"] for n in range(4)] == ["MAIN FLOOR PLAN", "UPPER FLOOR PLAN", "LOWER FLOOR PLAN", "SHEET 3"]


def test_title_block_with_no_number_leaves_it_blank():
    assert title_blocks({0: page(line("GENERAL NOTES", 2400, 1500))})[0]["number"] == ""


def test_title_block_box_is_in_points():
    got = title_blocks({0: page(*block("A1.0", "COVER SHEET"))})[0]["box"]
    assert got[0] == 2400 and got[1] == 1560


def test_drawing_labels_skip_the_title_block_and_small_text():
    lines = [line("Ground Floor Plan", 800, 1200, 18.9), line("Main Floor Plan", 2400, 1500, 18.9), line("see first floor plan", 100, 100, 9.6)]
    assert [x["t"] for x in drawing_labels(lines, W)] == ["Ground Floor Plan"]


@pytest.mark.parametrize("text, want", [("FIFTH FLOOR PLAN", {"5"}), ("5TH FLOOR PLAN", {"5"}), ("PLAN - LEVEL 5", {"5"}), ("TWELFTH FLOOR PLAN", {"12"}), ("LEVEL 12 PLAN", {"12"}),
                                        ("21ST FLOOR PLAN", {"21"}), ("LEVEL 05 PLAN", {"5"})])
def test_floor_words_name_every_storey(text, want):
    assert floor_words(text) == want
    assert [x["t"] for x in drawing_labels([line(text, 800, 1200, 18.9)], W)] == [text]


def test_index_rows_read_a_column_of_known_sheets():
    rows = [line(f"A{n}.0  Sheet {n}", 100, 200 + 20 * n, 18.9) for n in range(5)]
    got = cover_index(rows, {f"A{n}.0" for n in range(5)})
    assert [r["number"] for r in got] == [f"A{n}.0" for n in range(5)]
    assert got[1]["title"] == "Sheet 1"


def test_index_rows_read_a_number_and_a_title_in_two_cells():
    rows = []
    for n in range(4):
        rows += [line(f"A{n}.0", 100, 200 + 20 * n, 18.9, width=50), line(f"Sheet {n}", 170, 200 + 20 * n, 18.9)]
    assert [r["title"] for r in cover_index(rows, {f"A{n}.0" for n in range(4)})] == [f"Sheet {n}" for n in range(4)]


def test_index_ignores_a_list_of_code_sections():
    rows = [line(f"R30{n}.1  Smoke alarms", 100, 200 + 20 * n, 18.9) for n in range(5)]
    assert cover_index(rows, {"A1.0", "A2.0"}) == []


def test_not_used_row_flagged():
    rows = [line(f"A{n}.0  Sheet {n}", 100, 200 + 20 * n, 18.9) for n in range(4)] + [line("A4.0  (NOT USED)", 100, 280, 18.9)]
    got = cover_index(rows, {f"A{n}.0" for n in range(4)})
    assert [r["not_used"] for r in got] == [False] * 4 + [True]


def test_schedule_rows_keep_each_cell_under_its_column():
    got = schedules(DOORS)[0]
    assert (got["kind"], got["cols"]) == ("door", ["MARK", "SIZE", "SWING"])
    assert [r["MARK"] for r in got["rows"]] == ["1", "2", "3"]
    assert got["rows"][2]["SWING"] == "BIFOLD"


def test_schedule_cell_stays_under_the_text_cap_however_many_lines_share_it():
    stacked = [line("W" * 400, 240, 255 + 0.1 * n, 9.6, width=30) for n in range(50)]   # all on door 1's row, under SWING
    cells = [v for r in schedules(DOORS + stacked)[0]["rows"] for v in r.values() if isinstance(v, str)]
    assert max(map(len, cells)) == declared.MAX_TEXT


def test_room_compare_stays_under_the_text_cap(monkeypatch):
    seen, real = [], declared.difflib.SequenceMatcher
    monkeypatch.setattr(declared.difflib, "SequenceMatcher", lambda junk, a, b: seen.append(max(len(a), len(b))) or real(junk, a, b))
    room_same("a b " * 100000, "c d " * 100000)
    assert seen and max(seen) <= declared.MAX_TEXT


def test_schedule_with_no_header_row_is_not_read():
    assert schedules([line("DOOR SCHEDULE", 100, 200, 18.9), line("See sheet A6.0", 100, 240, 9.6)]) == []


@pytest.mark.parametrize("step", [760, 600, 400, 250, 210])   # a table here is 170 pt wide, so 210 leaves 40 pt between tables
def test_schedules_side_by_side_at_one_height_are_each_read_with_only_their_own_cells(step):
    level = [x for n, mark in enumerate(("101", "201", "301", "401", "501")) for x in table(f"DOOR SCHEDULE {n + 1}", ["MARK", "SIZE", "SWING"], [[mark, "36", "LEFT"], [f"{mark[0]}02", "34", "RIGHT"]], x=70 + step * n)]
    got = schedules(level)
    assert [[(r["MARK"], r["SIZE"]) for r in s["rows"]] for s in got] == [[(f"{n}01", "36"), (f"{n}02", "34")] for n in range(1, 6)]
    assert {tuple(s["cols"]) for s in got} == {("MARK", "SIZE", "SWING")}


def test_schedules_side_by_side_at_different_heights_end_where_the_first_column_is_printed_again():
    level = table("DOOR SCHEDULE", ["MARK", "SIZE", "SWING"], [["1", "36", "LEFT"]], x=70) + table("DOOR SCHEDULE", ["MARK", "SIZE", "SWING"], [["2", "34", "RIGHT"]], x=280, y=230)
    assert [(s["cols"], s["rows"][0]["MARK"], s["rows"][0]["SIZE"]) for s in schedules(level)] == [(["MARK", "SIZE", "SWING"], "1", "36"), (["MARK", "SIZE", "SWING"], "2", "34")]


def test_schedule_two_headings_name_is_read_once():
    assert len(schedules([*DOORS, line("SEE DOOR SCHEDULE", 100, 215, 9.6)])) == 1


def test_shape_tags_read_short_words_inside_small_shapes(tmp_path):
    def draw(p, i):
        s = p.new_shape()
        s.draw_circle((300, 300), 12)
        s.finish(color=(0, 0, 0))
        s.commit()
        p.insert_text((295, 304), "17", fontsize=9.6)
        p.insert_text((600, 300), "99", fontsize=9.6)   # a number with no shape round it is not a tag
    write_pdf(tmp_path / "a.pdf", draw)
    with pymupdf.open(tmp_path / "a.pdf") as doc:
        got = shape_tags(doc[0])
    assert [t[0] for t in got] == ["17"]
    assert got[0][3][2] == "circle"


def test_shape_tags_on_a_rotated_page_are_in_the_rendered_frame(tmp_path):
    def draw(p, i):
        s = p.new_shape()
        s.draw_circle((300, 300), 12)
        s.finish(color=(0, 0, 0))
        s.commit()
        p.insert_text((295, 304), "17", fontsize=9.6)
    write_pdf(tmp_path / "a.pdf", draw, rotation=90)
    with pymupdf.open(tmp_path / "a.pdf") as doc:
        got = shape_tags(doc[0])
    assert [t[0] for t in got] == ["17"]
    assert abs(got[0][1] - (H - 300)) < 12 and abs(got[0][2] - 300) < 12   # a 90 degree page puts x 300 at y 300 and y 300 at x = 1728 - 300


def test_plan_rooms_take_the_name_above_the_number_and_skip_tables():
    lines = [line("KITCHEN", 300, 300), line("102", 305, 320), line("102", 100, 220), line("BEDROOM", 100, 205)]
    got = plan_rooms(lines, [[90, 190, 200, 240]])
    assert [(r["no"], r["name"]) for r in got] == [("102", "KITCHEN")]


def test_plan_rooms_skip_a_number_inside_a_table_under_a_name_outside_it():
    lines = [line("HALL", 100, 170), line("103", 100, 195)]
    assert plan_rooms(lines, [[90, 190, 200, 240]]) == []
    assert [r["no"] for r in plan_rooms(lines, [])] == ["103"]


def test_plan_rooms_skip_a_sheet_number():
    lines = [line("GENERAL NOTES", 100, 170), line("A101", 100, 190)]
    assert plan_rooms(lines, [], {"A101"}) == []
    assert [r["no"] for r in plan_rooms(lines, [])] == ["A101"]


def test_area_table_reads_levels_and_the_total():
    lines = [line("Ground Level:", 750, 960), line("576 ft²", 900, 960), line("Main Level:", 750, 975), line("576 ft²", 900, 975),
             line("1,152 ft² (does not include slab)", 750, 990)]
    got = text_tables(lines)
    assert [t["kind"] for t in got] == ["area"]
    assert [(r["label"], r["area"]) for r in got[0]["rows"]] == [("Ground Level", 576.0), ("Main Level", 576.0), ("total", 1152.0)]


def occupant_lines():
    out = [line("OCCUPANCY CALCULATIONS", 2027, 83, 25.2)]
    for n, cells in enumerate((["100", "ENTRY", "701 SF", "15", "47"], ["101", "OFFICE", "109 SF", "150", "1"], ["103", "RESTROOM", "137 SF"], ["Grand total", "947 SF", "48"])):
        out += [line(c, 2000 + 80 * k, 314 + 13 * n, 9.5, width=40) for k, c in enumerate(cells)]
    return out


def test_occupant_table_reads_rooms_and_the_total():
    got = text_tables(occupant_lines())
    assert [t["kind"] for t in got] == ["occupant"]
    rows = got[0]["rows"]
    assert rows[0] == {"room": "100", "name": "ENTRY", "area": 701.0, "factor": 15.0, "occupants": 47, "box": rows[0]["box"]}
    assert (rows[2]["factor"], rows[2]["occupants"]) == (None, None)
    assert (rows[3]["name"], rows[3]["area"], rows[3]["occupants"], rows[3]["total"]) == ("Grand total", 947.0, 48, True)


def test_exit_table_reads_rows_and_the_worked_example():
    lines = [line("EXIT WIDTH CALCULATION", 2027, 1536, 25.2)]
    for n, cells in enumerate((["EXIT 1", "36", "33", "90", "18", "OK"], ["EXIT 2", "36", "33", "96", "5", "19.2", "OK"])):
        lines += [line(c, 2000 + 50 * k, 1847 + 13 * n, 9.5, width=30) for k, c in enumerate(cells)]
    lines += [line("EXIT 1:", 2000, 1951, 9.6), line('100 OCUPANTS * 0.2" = 20.0" REQUIRED (X)', 2000, 1962, 9.6)]
    rows = text_tables(lines)[0]["rows"]
    assert [(r["exit"], r["clear_in"], r["occupants"], r["required_in"]) for r in rows[:2]] == [("EXIT 1", 33.0, 90, 18.0), ("EXIT 2", 33.0, 96, 19.2)]
    assert (rows[2]["exit"], rows[2]["occupants"], rows[2]["required_in"], rows[2]["example"]) == ("EXIT 1", 100, 20.0, True)


@pytest.mark.parametrize("under", [[], [line(c, 2000 + 80 * k, 314, 9.5, width=40) for k, c in enumerate(["101", "OFFICE", "109 SQ FT", "150", "1"])]],   # no pattern reads SQ FT
                         ids=["table-drawn-as-lines", "row-no-pattern-reads"])
def test_table_heading_with_no_rows_is_returned_empty_for_cannot_tell(under):
    got = text_tables([line("OCCUPANCY CALCULATIONS", 2027, 83, 25.2), *under])
    assert [(t["kind"], t["rows"]) for t in got] == [("occupant", [])]


@pytest.mark.parametrize("lines", [
    [line("OCCUPANT LOAD:", 70, 440, 16), line("90", 290, 440, 16)],
    [line("OCCUPANT LOAD THIS FLOOR", 1590, 96, 16), line("GROSS FLOOR AREA", 1590, 126, 11), line("3,401 SF", 1830, 126, 11), line("OCCUPANT LOAD", 1590, 148, 11),
     line("18 PEOPLE", 1830, 148, 11)],
    [line("OCCUPANCY CALCULATIONS", 2027, 83, 25.2), line("See sheet A2", 2000, 200, 9.5)],
], ids=["label-and-value", "labels-and-values", "note"])
def test_heading_with_no_table_under_it_is_no_table(lines):
    assert text_tables(lines) == []


def test_a_column_head_standing_on_end_is_not_a_table_heading():
    assert text_tables([line("Occupant Load Factor", 2325, 105, 19.0, along=False)]) == []


def test_text_reads_give_each_page_its_title_index_and_tables():
    pages = {0: page(*block("A1.0", "COVER SHEET"), *[line(f"A{n}.0  Sheet {n}", 100, 200 + 20 * n, 18.9) for n in range(1, 5)]),
             1: page(*block("A1.0", "PLAN"), *DOORS)}
    got = text_reads(pages)
    assert got[1]["schedules"][0]["kind"] == "door"
    assert got[0]["chars"] > 0


def test_title_of_a_hostile_strip_is_capped():
    lines = [line(f"NOTE {n} " + "x" * 100, 2400, 100 + n) for n in range(300)]
    assert len(title_blocks({0: page(line("A1.0", 2420, 1620, 31.5), *lines)})[0]["title"]) == declared.MAX_TEXT


def test_page_whose_tables_cannot_be_parsed_keeps_its_title_and_tags_and_costs_no_other_page(monkeypatch):
    pages = {0: page(*block("A1.0", "COVER SHEET"), *DOORS), 1: page(*block("A2.1", "PLAN"), *DOORS, line("ODD", 100, 900), tags=tagged("1"))}
    real = declared.schedules
    monkeypatch.setattr(declared, "schedules", lambda lines: 1 / 0 if any(x["t"] == "ODD" for x in lines) else real(lines))
    got = text_reads(pages)
    assert got[1]["unread"] and got[1]["schedules"] == [] and got[1]["tables"] == []
    assert (got[1]["number"], got[1]["tags"], got[1]["chars"] > 0) == ("A2.1", tagged("1"), True)
    assert "unread" not in got[0] and got[0]["schedules"]


# ---- the stage on a real PDF, through the child process ----

def text_set(p, i):
    p.insert_text((2400, 1560), ["COVER SHEET", "MAIN FLOOR PLAN"][i], fontsize=12.6)
    p.insert_text((2420, 1620), ["A1.0", "A2.1"][i], fontsize=31.5)
    if i == 0:
        p.insert_text((751, 904), "2015 Michigan Residential Code", fontsize=12.6)
        for n, row in enumerate(("A1.0  Cover", "A2.1  Main Floor Plan", "A3.0  Sections", "A4.0  Elevations")):
            p.insert_text((100, 200 + 22 * n), row, fontsize=18.9)
    else:
        p.insert_text((800, 1200), "Ground Floor Plan", fontsize=18.9)
        for x in table("DOOR SCHEDULE", ["MARK", "SIZE", "SWING"], [["1", "36", "LEFT"], ["2", "36", "RIGHT"]]):
            p.insert_text((x["x0"], x["y0"] + x["size"]), x["t"], fontsize=x["size"])
        s = p.new_shape()
        for c in ((300, 900), (500, 900)):
            s.draw_circle(c, 12)
            s.finish(color=(0, 0, 0))   # one finish per circle, or the two read as one wide shape
        s.commit()
        p.insert_text((296, 904), "1", fontsize=9.6)
        p.insert_text((496, 904), "2", fontsize=9.6)
        p.insert_text((1000, 700), "KITCHEN", fontsize=12.6)
        p.insert_text((1010, 720), "102", fontsize=12.6)


def text_rows():
    return [read_row(0, "A1.0", "COVER SHEET", "cover"), read_row(1, "A2.1", "MAIN FLOOR PLAN", "floor_plan")]


def test_text_layer_used_before_model(tmp_path):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    model = Model({"This image is one sheet of a building drawing set. Read what it states": FIELDS, "Two strings": {"results": []}})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert all(p.startswith("This image is one sheet of a building drawing set. Read what it states") for p in model.calls)
    assert [(t["number"], t["title"], t["source"]) for t in out["titles"]] == [("A1.0", "COVER SHEET", "text"), ("A2.1", "MAIN FLOOR PLAN", "text")]
    assert out["titles"][1]["under_drawing"] == "Ground Floor Plan"
    assert [r["number"] for r in out["index"]] == ["A1.0", "A2.1", "A3.0", "A4.0"]
    assert out["index"][0]["sheet"] == 0
    door = out["schedules"][0]
    assert (door["sheet"], door["kind"], door["source"]) == (1, "door", "text")
    assert [r["mark"] for r in door["rows"]] == ["1", "2"]
    assert door["rows"][1]["cells"] == {"SIZE": "36", "SWING": "RIGHT"}
    assert [(t["sheet"], t["kind"], t["mark"]) for t in out["tags"]] == [(1, "door", "1"), (1, "door", "2")]
    json.dumps(out, allow_nan=False)


def test_boxes_and_points_are_sheet_pixels_not_page_points(tmp_path):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    meta.set_transport(Model({"This image is one sheet": FIELDS}))
    out = declared_stage(ctx)
    x, y = out["tags"][0]["point"]
    assert abs(x - 300 * PX) < 12 and abs(y - 900 * PX) < 12
    box = out["schedules"][0]["rows"][0]["box"]
    assert box[0] == pytest.approx(100 * PX, abs=4)
    assert out["code"]["edition"]["box"] is not None and out["code"]["edition"]["box"][0] > 751


def test_pdf_is_read_in_a_child_process(tmp_path, monkeypatch):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    monkeypatch.setattr(pymupdf, "open", lambda *a, **k: pytest.fail("the stage opened the PDF itself"))
    meta.set_transport(Model({"This image is one sheet": FIELDS}))
    assert declared_stage(ctx)["index"]


def test_stuck_child_is_killed_and_the_stage_finishes(tmp_path, monkeypatch):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    monkeypatch.setattr(declared, "COMMAND", [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(declared, "CHILD_SECONDS", 0.5)
    meta.set_transport(Model({"This image is one sheet": FIELDS}))
    out = declared_stage(ctx)
    assert any("text layer" in row["why"] for row in out["cannot_tell"])
    assert [t["source"] for t in out["titles"]] == ["spark", "spark"]


def test_rerun_never_reads_an_earlier_childs_files(tmp_path, monkeypatch):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    meta.set_transport(Model({"This image is one sheet": FIELDS}))
    assert [t["source"] for t in declared_stage(ctx)["titles"]] == ["text", "text"]
    monkeypatch.setattr(declared, "COMMAND", [sys.executable, "-c", "pass"])   # a child that reads nothing
    assert [t["source"] for t in declared_stage(ctx)["titles"]] == ["spark", "spark"]


def test_a_value_a_browser_cannot_parse_is_refused_and_leaves_no_file(tmp_path):
    with pytest.raises(ValueError):
        declared.save(tmp_path / "a.json", {"x": float("inf")})
    assert not (tmp_path / "a.json").exists()


def test_every_loop_over_the_uploads_text_runs_in_the_child(tmp_path, monkeypatch):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    monkeypatch.setattr(declared, "text_reads", lambda pages: pytest.fail("the stage parsed the upload's text itself"))
    monkeypatch.setattr(declared, "select", lambda *a: pytest.fail("the stage paired schedules and plans itself"))
    meta.set_transport(Model({"This image is one sheet": FIELDS}))
    assert declared_stage(ctx)["tags"]


def test_child_is_given_the_read_rows_and_floors_it_selects_with(tmp_path):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    write_json(tmp_path / "stage_floors.json", {"order": [1]})
    meta.set_transport(Model({"This image is one sheet": FIELDS}))
    declared_stage(ctx)
    job = json.loads((tmp_path / "declared/job.json").read_text())
    assert job["order"] == [1] and [r["index"] for r in job["rows"]] == [0, 1]


def test_selection_is_plain_json_for_the_child_to_hand_over():
    pages = {1: page(*block("A2.1", "MAIN FLOOR PLAN"), *DOORS, tags=tagged("1", "2", "3"))}
    assert json.loads(json.dumps(picks([read_row(1, "A2.1", "MAIN FLOOR PLAN", "floor_plan")], pages)))


def test_sheet_with_many_schedules_is_capped():
    tables = [x for n in range(30) for x in table("DOOR SCHEDULE", ["MARK", "SIZE", "SWING"], [["1", "36", "LEFT"]], x=100 + 1000 * (n % 2), y=100 + 200 * n)]
    assert len(schedules(tables)) == 30
    assert len(text_reads({0: page(*tables)})[0]["schedules"]) == declared.MAX_SCHEDULES


def test_page_that_cannot_be_read_costs_only_itself(tmp_path, monkeypatch):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    real = declared.page_lines
    job = {"pdf": str(tmp_path / "upload/0000.pdf"), "sheets": [0, 1], "rows": [], "order": None}
    (tmp_path / "work").mkdir()
    monkeypatch.setattr(declared, "page_lines", lambda p: 1 / 0 if p.number == 0 else real(p))
    declared.parse_all(job, tmp_path / "work")
    assert "failed" in json.loads((tmp_path / "work/0.json").read_text())
    assert json.loads((tmp_path / "work/1.json").read_text())["lines"]


def names_only(p, i):
    p.insert_text((2400, 1560), ["SCHEDULES", "FIRST FLOOR PLAN"][i], fontsize=12.6)
    p.insert_text((2420, 1620), ["A6.0", "A2.1"][i], fontsize=31.5)
    if i == 0:
        for x in table("FINISH SCHEDULE", ["NO.", "NAME", "FLOOR"], [["101", "KITCHEN", "TILE"], ["102", "PANTRY", "TILE"]]):
            p.insert_text((x["x0"], x["y0"] + x["size"]), x["t"], fontsize=x["size"])
    else:
        p.insert_text((1000, 700), "KITCHEN", fontsize=12.6)
        p.insert_text((1400, 700), "PANTRY", fontsize=12.6)


def names_only_rows():
    return [read_row(0, "A6.0", "SCHEDULES", "schedule"), read_row(1, "A2.1", "FIRST FLOOR PLAN", "floor_plan")]


def test_plan_that_prints_names_only_gives_the_schedule_names_it_holds(tmp_path):
    ctx = pdf_review(tmp_path, names_only, names_only_rows(), pages=2)
    meta.set_transport(Model({}))
    assert declared_stage(ctx)["rooms"] == [{"sheet": 1, "number": "", "name": n, "source": "text", "box": None} for n in ("KITCHEN", "PANTRY")]


def test_names_looked_for_on_a_plan_are_capped(tmp_path, monkeypatch):
    ctx = pdf_review(tmp_path, names_only, names_only_rows(), pages=2)
    monkeypatch.setattr(declared, "MAX_NAMES", 1)
    meta.set_transport(Model({}))
    assert [r["name"] for r in declared_stage(ctx)["rooms"]] == ["KITCHEN"]


def test_labels_under_a_drawing_are_capped(tmp_path):
    def draw(p, i):
        p.insert_text((2400, 1560), "FLOOR PLANS", fontsize=12.6)
        p.insert_text((2420, 1620), "A2.1", fontsize=31.5)
        for n in range(40):
            p.insert_text((100, 100 + 30 * n), f"FIRST FLOOR PLAN PART {n}", fontsize=18.9)
    ctx = pdf_review(tmp_path, draw, [read_row(0, "A2.1", "FLOOR PLANS", "floor_plan")])
    meta.set_transport(Model({}))
    assert len(declared_stage(ctx)["titles"][0]["under_drawing"]) == declared.MAX_TEXT


def child_where(patch):
    """The child with `patch` applied to the module `d` first, for a fault that only the child can meet."""
    return [sys.executable, "-c", f"import sys\nimport storey.declared as d\n{patch}\nsys.exit(d.main(sys.argv[1:]))"]


PAGE_1_FAILS = "real = d.page_lines\nd.page_lines = lambda p: 1 / 0 if p.number == 1 else real(p)"


def schedule_and_plans(*headings):
    """Page 0 holds the schedules. Every later page is a plan of the same sheet letter."""
    def draw(p, i):
        p.insert_text((2400, 1560), "SCHEDULES" if i == 0 else f"FLOOR {i} PLAN", fontsize=12.6)
        p.insert_text((2420, 1620), f"A{i}.0", fontsize=31.5)
        for n, heading in enumerate(headings if i == 0 else ()):
            for x in table(heading, ["MARK", "SIZE", "TYPE"], [["1", "36", "A"], ["2", "48", "B"]], x=100 + 900 * n):
                p.insert_text((x["x0"], x["y0"] + x["size"]), x["t"], fontsize=x["size"])
        if i > 1:   # the tags of a plan the fault leaves alone
            s = p.new_shape()
            s.draw_circle((300, 900), 12)
            s.finish(color=(0, 0, 0))
            s.commit()
            p.insert_text((296, 904), "1", fontsize=9.6)
    return draw


def schedule_rows(pages):
    return [read_row(0, "A0.0", "SCHEDULES", "schedule"), *[read_row(i, f"A{i}.0", f"FLOOR {i} PLAN", "floor_plan", f"LEVEL {i}") for i in range(1, pages)]]


def test_plan_the_child_could_not_read_beside_a_finish_schedule_costs_only_itself(tmp_path, monkeypatch):
    ctx = pdf_review(tmp_path, schedule_and_plans("FINISH SCHEDULE"), schedule_rows(3), pages=3)
    monkeypatch.setattr(declared, "COMMAND", child_where(PAGE_1_FAILS))
    meta.set_transport(Model({TB_START: block_read("A1.0", "FLOOR 1 PLAN")}))
    out = declared_stage(ctx)
    assert [t["source"] for t in out["titles"]] == ["text", "spark", "text"]
    assert any(row["sheet"] == 1 and "text layer" in row["why"] for row in out["cannot_tell"])


def test_only_plan_of_a_finish_schedule_lost_with_the_child_goes_to_spark(tmp_path, monkeypatch):
    ctx = pdf_review(tmp_path, schedule_and_plans("FINISH SCHEDULE"), schedule_rows(2), pages=2)
    write_json(tmp_path / "stage_floors.json", {"order": [1]})
    monkeypatch.setattr(declared, "COMMAND", child_where(PAGE_1_FAILS))
    room = {"rooms": [{"number": "1E08", "name": "OFFICE", "x": 500, "y": 500}]}
    model = Model({TB_START: block_read("A1.0", "FLOOR 1 PLAN"), ROOMS: room})
    meta.set_transport(model)
    assert [r["number"] for r in declared_stage(ctx)["rooms"]] == ["1E08"]
    assert model.count(ROOMS) == 18


def test_plan_the_child_could_not_read_beside_a_door_schedule_is_named(tmp_path, monkeypatch):
    # the row exists because Task 9 would otherwise call a row whose tag sits on the lost plan untagged
    ctx = pdf_review(tmp_path, schedule_and_plans("DOOR SCHEDULE"), schedule_rows(3), pages=3)
    monkeypatch.setattr(declared, "COMMAND", child_where(PAGE_1_FAILS))
    meta.set_transport(Model({TB_START: block_read("A1.0", "FLOOR 1 PLAN")}))
    out = declared_stage(ctx)
    assert out["schedules"][0]["plans"] == [2]
    assert any(row["sheet"] == 1 and "text layer" in row["why"] for row in out["cannot_tell"])


def test_page_whose_tables_fail_in_the_child_says_so_and_keeps_the_rest_of_the_text_layer(tmp_path, monkeypatch):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    patch = "real = d.schedules\nd.schedules = lambda lines: 1 / 0 if any('KITCHEN' in x['t'] for x in lines) else real(lines)"
    monkeypatch.setattr(declared, "COMMAND", child_where(patch))
    model = Model({CODE_START: FIELDS})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert len(out["index"]) == 4 and [t["source"] for t in out["titles"]] == ["text", "text"]
    assert out["schedules"] == []
    assert any(row["what"] == "Text layer" and row["sheet"] == 1 for row in out["cannot_tell"])


def odd_table(row):
    def draw(p, i):
        p.insert_text((2400, 1560), ["COVER SHEET", "MAIN FLOOR PLAN"][i], fontsize=12.6)
        p.insert_text((2420, 1620), ["A1.0", "A2.1"][i], fontsize=31.5)
        if i == 0:
            for n, r in enumerate(("A1.0  Cover Sheet", "A2.1  Main Floor Plan", "A3.0  Sections", "A4.0  Elevations")):
                p.insert_text((100, 200 + 22 * n), r, fontsize=18.9)
        else:
            p.insert_text((100, 200), "EXIT WIDTH CALCULATION", fontsize=25.2)
            p.insert_text((100, 300), row, fontsize=9.6)
    return draw


@pytest.mark.parametrize("row, key", [
    ("EXIT 1 36 33 90 1.2.3 OK", "required_in"),
    ('100 OCUPANTS * 0.2" = 20.0. REQUIRED', "required_in"),
    ('101 HALL 90 18 33 1/0" OK', "clear_in"),
    (f"EXIT 1 36 {'9' * 330} 90 18 OK", "clear_in"),
], ids=["two-dots", "trailing-dot", "zero-denominator", "too-long-for-a-float"])
def test_one_odd_table_row_costs_that_number_and_not_the_text_layer(tmp_path, row, key):
    ctx = pdf_review(tmp_path, odd_table(row), text_rows(), pages=2)
    model = Model({CODE_START: FIELDS})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert len(out["index"]) == 4 and [t["source"] for t in out["titles"]] == ["text", "text"]
    assert out["tables"][0]["rows"][0][key] is None
    assert any(row["what"] == "Exit table" and row["sheet"] == 1 for row in out["cannot_tell"])
    json.dumps(out, allow_nan=False)


def test_area_too_long_for_a_float_reads_as_no_area():
    lines = [line("Ground Level:", 750, 960), line("9" * 400 + " ft²", 900, 960, 3), line("1,152 ft² (does not include slab)", 750, 990)]
    assert text_tables(lines)[0]["rows"][0]["area"] is None


def test_stage_module_leaves_the_pipeline_free_to_register_it():
    # Task 11 imports every stage from pipeline.py, and a stage importing pipeline back stops the app from starting
    check = "import sys, storey.declared; sys.exit('storey.pipeline' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", check], cwd=declared.ROOT).returncode == 0


def test_image_upload_reads_with_spark_and_never_starts_the_child(tmp_path, monkeypatch):
    ctx = review(tmp_path, [read_row(0, "A1.0", "COVER SHEET", "cover")])
    monkeypatch.setattr(declared, "COMMAND", ["/nonexistent"])
    tb = {"sheet_number": "A1.0", "sheet_title": "COVER SHEET", "sequence": ""}
    meta.set_transport(Model({TB_START: tb, CODE_START: FIELDS}))
    assert declared_stage(ctx)["titles"][0]["source"] == "spark"


def test_no_read_stage_still_reads_the_text_layer(tmp_path):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    (tmp_path / "stage_read.json").unlink()
    meta.set_transport(Model({"This image is one sheet": FIELDS}))
    assert len(declared_stage(ctx)["index"]) == 4


# ---- two reads ----

@pytest.mark.parametrize("a, b, want", [
    ("1,728 SF", "1,728 SF", "1,728 SF"),
    ("1,728 SF", "1728 SF", "1728 SF"),
    ("International Building Code, 2012 Edition.", "International Building Code, 2012 Edition", "International Building Code, 2012 Edition"),
    ("2015 Michigan Residential Code", "2015 Michigan Residential Code\nMichigan Barrier Free Codes", None),
    ("BUSINESS Group: B", "BUSINESS", None),
    ("Area of Work (Total): 4,351; 3d Floor Total Area: 16,350", "4,351", None),
    ("2021 IBC", "2015 IBC, 2021 IFC", None),
    ("R-2", "R-2, A-2", None),
    ("R-2", "R-2.1", None),
    ("2", "2 STORIES AND BASEMENT, 3 LEVELS", None),
    ("1.5", "5.1", None),
    ("9,373 SF", "9,273 SF", None),
    ("367 OCCUPANTS", "357 OCCUPANTS", None),
    ("1,728 SF", "728 SF", None),
    ("", "1,728 SF", None),
    ("", "", None),
    ("-", "-", None),
], ids=["same", "thousands-comma-dropped", "full-stop", "added-name", "added-word", "added-area", "added-edition", "added-group", "subgroup", "added-stories", "digits-swapped", "digit", "load", "thousands-comma", "one-empty", "both-empty", "both-punctuation"])
def test_two_reads_agree_only_on_the_same_words(a, b, want):
    assert agree(a, b) == want


def test_two_reads_disagree_gives_cannot_tell():
    a, b = FIELDS | {"area": "9,373 SF"}, FIELDS | {"area": "9,273 SF"}
    code, notes = code_fields({0: [a, b]})
    assert code["area"] == {"value": None, "sheet": 0, "box": None, "reads": 0}
    assert code["occupancy"] == {"value": "R-3", "sheet": 0, "box": None, "reads": 2, "mentions": []}
    assert [n["sheet"] for n in notes] == [0]
    assert "9,373 SF" in notes[0]["why"] and "9,273 SF" in notes[0]["why"]


def test_field_both_reads_leave_blank_is_not_cannot_tell():
    code, notes = code_fields({0: [FIELDS | {"stories": ""}, FIELDS | {"stories": ""}]})
    assert code["stories"] == {"value": None, "sheet": 0, "box": None, "reads": 2}
    assert notes == []


def test_not_stated_sprinklers_read_as_no_value():
    code, notes = code_fields({0: [FIELDS | {"sprinklered": "not_stated"}] * 2})
    assert code["sprinklers"]["value"] is None and code["sprinklers"]["reads"] == 2


def test_later_code_sheet_fills_a_field_the_first_leaves_blank():
    code, _ = code_fields({0: [FIELDS | {"occupancy": ""}] * 2, 5: [FIELDS | {"occupancy": "BUSINESS"}] * 2})
    assert (code["occupancy"]["value"], code["occupancy"]["sheet"]) == ("BUSINESS", 5)
    assert code["construction"]["sheet"] == 0


def test_lost_read_on_a_later_code_sheet_outranks_a_blank_on_the_first():
    # both reads of the first sheet found nothing printed, so the later sheet is the only one that could hold the field
    code, notes = code_fields({0: [FIELDS | {"area": ""}] * 2, 5: [FIELDS | {"area": "100 SF"}, None]})
    assert code["area"] == {"value": None, "sheet": 5, "box": None, "reads": 1}
    assert any(n["what"] == "Code field area" and n["sheet"] == 5 for n in notes)


def test_lost_read_is_cannot_tell_for_every_field():
    code, notes = code_fields({0: [FIELDS, None]})
    assert code["area"] == {"value": None, "sheet": 0, "box": None, "reads": 1}
    assert notes


def test_malformed_code_reply_counts_as_lost():
    code, notes = code_fields({0: [{"occupancy": 7}, FIELDS]})
    assert code["occupancy"]["value"] is None and notes


def test_missing_code_sheet_returns_empty_code(tmp_path):
    ctx = review(tmp_path, [read_row(0, "A2.1", "FLOOR PLAN", "floor_plan")])
    meta.set_transport(Model({TB_START: {"sheet_number": "A2.1", "sheet_title": "FLOOR PLAN", "sequence": ""}}))
    out = declared_stage(ctx)
    assert out["code"] == {}
    assert any(row["what"] == "Code summary" for row in out["cannot_tell"])


def test_edition_with_no_year_comes_from_the_text_layer(tmp_path):
    def draw(p, i):
        p.insert_text((2400, 1560), "TITLE SHEET", fontsize=12.6)
        p.insert_text((2420, 1620), "G-001", fontsize=31.5)
        p.insert_text((100, 300), "OCCUPANCY GROUP: R-3", fontsize=9.6)
        p.insert_text((100, 320), "2022 CBC APPLIES", fontsize=9.6)
    ctx = pdf_review(tmp_path, draw, [read_row(0, "G-001", "TITLE SHEET", "cover")])
    meta.set_transport(Model({"This image is one sheet": FIELDS | {"code_edition": "CURRENT CALIFORNIA BUILDING CODE"}}))
    got = declared_stage(ctx)["code"]["edition"]
    assert (got["value"], got["reads"]) == ("2022 CBC", 0)


def test_edition_the_reads_differ_on_comes_from_the_text_layer_with_no_row(tmp_path):
    # Capitol's and Baxter's reads differ by a second code name, and the text layer settles the edition
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    codes = [FIELDS | {"code_edition": "2015 Michigan Residential Code"}, FIELDS | {"code_edition": "2015 Michigan Residential Code Michigan Barrier Free Codes"}]
    meta.set_transport(Model({CODE_START: codes}))
    out = declared_stage(ctx)
    assert (out["code"]["edition"]["value"], out["code"]["edition"]["reads"]) == ("2015 Michigan Residential Code", 0)
    assert not any("edition" in row["what"] for row in out["cannot_tell"])


def test_mentions_both_reads_give_stay_on_occupancy_and_sprinklers():
    # USC G002: the sheet says no system and also places sprinkler heads, and a table calls the floor Institutional Outpatient
    a = FIELDS | {"sprinkler_mentions": ["SPRINKLER HEADS SHALL BE LOCATED IN CENTER ZONE.", "D. is the building provided with a fire protection sprinkler system? NO"], "occupancy_mentions": ["Institutional Outpatient B"]}
    b = FIELDS | {"sprinkler_mentions": ["Is the building provided with a fire protection sprinkler system? NO", "SPRINKLER HEADS SHALL BE LOCATED IN CENTER ZONE"], "occupancy_mentions": ["B", "Institutional Outpatient B"]}
    code, _ = code_fields({0: [a, b]})
    assert code["sprinklers"]["mentions"] == ["SPRINKLER HEADS SHALL BE LOCATED IN CENTER ZONE."]
    assert code["occupancy"]["mentions"] == ["Institutional Outpatient B"]
    assert "mentions" not in code["area"]


def test_mentions_come_from_the_sheet_the_field_rests_on():
    first = FIELDS | {"occupancy": "", "occupancy_mentions": ["ON THE FIRST SHEET"]}
    later = FIELDS | {"occupancy": "BUSINESS", "occupancy_mentions": ["ON THE LATER SHEET"]}
    assert code_fields({0: [first, first], 5: [later, later]})[0]["occupancy"]["mentions"] == ["ON THE LATER SHEET"]


@pytest.mark.parametrize("bad", ["no", [7, None], None], ids=["string", "wrong-items", "missing"])
def test_mentions_of_the_wrong_shape_count_as_none(bad):
    code, _ = code_fields({0: [FIELDS | {"sprinkler_mentions": ["FULLY SPRINKLERED"]}, FIELDS | {"sprinkler_mentions": bad}]})
    assert code["sprinklers"]["mentions"] == []


def test_lost_read_leaves_no_mentions():
    assert code_fields({0: [FIELDS | {"occupancy_mentions": ["B"]}, None]})[0]["occupancy"]["mentions"] == []


# ---- exit capacity ----

CLIPS = tiles(2, 2)


def tile(*markers):
    return {"exits": [dict(zip(("x", "y", "width", "capacity"), m)) for m in markers], "travel_distances": [], "occupant_loads": []}


EMPTY = tile()


def at(clip, x, y):
    """Where a sheet point, as a 0-1 share of the sheet, lands on a tile's 0-1000 grid."""
    return round((x - clip[0]) / (clip[2] - clip[0]) * 1000), round((y - clip[1]) / (clip[3] - clip[1]) * 1000)


def test_half_sheet_rule_keeps_marker_from_its_own_tile():
    # 0.52 lies in the overlap of the two top tiles and in the right half, so only the right tile may count it. The other two sit 12 per mille apart.
    left, right = CLIPS[0], CLIPS[2]
    reply = [tile((*at(left, 0.52, 0.2), "915", 180)), EMPTY, tile((*at(right, 0.52, 0.2), "915", 180), (*at(right, 0.70, 0.2), "915", 180), (*at(right, 0.712, 0.2), "915", 180)), EMPTY]
    row, notes = exit_sheet(4, (6000, 4000), CLIPS, [reply, reply])
    assert row["sum"] == 540
    assert len(row["markers"]) == 3
    assert notes == []


def test_marker_points_and_widths_are_converted():
    reply = [tile((*at(CLIPS[0], 0.25, 0.25), "915", 180)), EMPTY, EMPTY, EMPTY]
    row, _ = exit_sheet(4, (6000, 4000), CLIPS, [reply, reply])
    x, y = row["markers"][0]["point"]
    assert abs(x - 1500) < 8 and abs(y - 1000) < 8
    assert row["markers"][0]["width_mm"] == 915
    assert row["markers"][0]["capacity"] == 180


def test_sum_needs_two_equal_reads():
    a = [tile((*at(CLIPS[0], 0.25, 0.25), "915", 180)), EMPTY, EMPTY, EMPTY]
    b = [tile((*at(CLIPS[0], 0.25, 0.25), "915", 180), (*at(CLIPS[0], 0.4, 0.25), "915", 180)), EMPTY, EMPTY, EMPTY]
    row, notes = exit_sheet(4, (6000, 4000), CLIPS, [a, b])
    assert row["sum"] is None and row["reads"] == [180, 360]
    assert notes[0]["sheet"] == 4 and "180" in notes[0]["why"] and "360" in notes[0]["why"]
    assert len(row["markers"]) == 1   # the marker both reads saw still stands for routes


@pytest.mark.parametrize("bad", [None, {}, {"exits": [{"x": 1}]}, {"exits": "no"}], ids=["no-reply", "empty", "missing-keys", "wrong-type"])
def test_empty_tile_reply_is_cannot_tell(bad):
    good = [tile((*at(CLIPS[0], 0.25, 0.25), "915", 180)), EMPTY, EMPTY, EMPTY]
    row, notes = exit_sheet(4, (6000, 4000), CLIPS, [good, [good[0], bad, EMPTY, EMPTY]])
    assert row["sum"] is None and row["reads"] is None
    assert notes and notes[0]["sheet"] == 4


def test_pull_station_with_no_capacity_or_width_is_dropped():
    reply = [tile((*at(CLIPS[0], 0.25, 0.25), "", 0), (*at(CLIPS[0], 0.1, 0.1), "915", 180)), EMPTY, EMPTY, EMPTY]
    row, _ = exit_sheet(4, (6000, 4000), CLIPS, [reply, reply])
    assert len(row["markers"]) == 1 and row["sum"] == 180


def test_marker_with_a_width_only_stays_and_adds_nothing_to_the_sum():
    reply = [tile((*at(CLIPS[0], 0.25, 0.25), "36 in", 0), (*at(CLIPS[0], 0.1, 0.1), "", 147)), EMPTY, EMPTY, EMPTY]
    row, _ = exit_sheet(4, (6000, 4000), CLIPS, [reply, reply])
    assert sorted(m["capacity"] or 0 for m in row["markers"]) == [0, 147]
    assert row["sum"] == 147


def test_sheet_with_no_capacity_printed_has_no_sum_and_is_cannot_tell():
    row, notes = exit_sheet(4, (6000, 4000), CLIPS, [[EMPTY] * 4, [EMPTY] * 4])
    assert row["sum"] is None and row["reads"] == [0, 0]
    assert notes


def test_marker_on_a_grid_past_the_tile_is_not_trusted():
    reply = [EMPTY, EMPTY, tile((1100, 200, "915", 180)), EMPTY]   # the top right tile, whose half of the sheet would otherwise hold it
    row, _ = exit_sheet(4, (6000, 4000), CLIPS, [reply, reply])
    assert row["markers"] == []


def test_coordinate_too_long_to_divide_is_dropped_and_no_crash():
    reply = [tile((10 ** 400, 200, "915", 180)), EMPTY, EMPTY, EMPTY]
    assert exit_sheet(4, (6000, 4000), CLIPS, [reply, reply])[0]["markers"] == []


@pytest.mark.parametrize("x, width, capacity", [(0.25, "915", 147), (0.25, "1830", 180), (0.30, "915", 180)], ids=["capacity", "width", "place"])
def test_marker_needs_both_reads_to_give_the_same_one(x, width, capacity):
    a = [tile((*at(CLIPS[0], 0.25, 0.25), "915", 180)), EMPTY, EMPTY, EMPTY]
    b = [tile((*at(CLIPS[0], x, 0.25), width, capacity)), EMPTY, EMPTY, EMPTY]
    assert exit_sheet(4, (6000, 4000), CLIPS, [a, b])[0]["markers"] == []


def test_marker_two_reads_place_close_together_is_one_marker():
    a = [tile((*at(CLIPS[0], 0.25, 0.25), "915", 180)), EMPTY, EMPTY, EMPTY]
    b = [tile((*at(CLIPS[0], 0.28, 0.25), "915", 180)), EMPTY, EMPTY, EMPTY]
    assert len(exit_sheet(4, (6000, 4000), CLIPS, [a, b])[0]["markers"]) == 1


@pytest.mark.parametrize("width", ["N/A", "-", "WIDE", "2 x 915"])
def test_marker_with_no_number_to_read_is_dropped(width):
    reply = [tile((*at(CLIPS[0], 0.25, 0.25), width, 0), (*at(CLIPS[0], 0.1, 0.1), "915", 180)), EMPTY, EMPTY, EMPTY]
    row, _ = exit_sheet(4, (6000, 4000), CLIPS, [reply, reply])
    assert len(row["markers"]) == 1 and row["markers"][0]["capacity"] == 180


def test_two_reads_that_give_one_width_with_and_without_its_words_pair():
    a = [tile((*at(CLIPS[0], 0.25, 0.25), '46" CLEAR WIDTH', 153)), EMPTY, EMPTY, EMPTY]
    b = [tile((*at(CLIPS[0], 0.25, 0.25), '46"', 153)), EMPTY, EMPTY, EMPTY]
    assert [(m["width_mm"], m["capacity"]) for m in exit_sheet(4, (6000, 4000), CLIPS, [a, b])[0]["markers"]] == [(pytest.approx(1168.4), 153)]


def test_marker_with_a_capacity_stays_when_its_width_cannot_be_read():
    reply = [tile((*at(CLIPS[0], 0.25, 0.25), "2 x 915", 180)), EMPTY, EMPTY, EMPTY]
    row, _ = exit_sheet(4, (6000, 4000), CLIPS, [reply, reply])
    assert [(m["width_mm"], m["capacity"]) for m in row["markers"]] == [(None, 180)]


@pytest.mark.parametrize("printed, want", [
    ("915", 915), ("36", 914.4), ("915 MM", 915), ("1830mm", 1830), ("1,830 mm", 1830), ('36"', 914.4), ("36\u201d", 914.4), ("36 in", 914.4), ("36 IN.", 914.4), ("36 inches", 914.4), ("3'-0\"", 914.4),
    ("3'", 914.4), ("3\u2019", 914.4), ("6 FT", 1828.8), ("6 feet", 1828.8), ("0.915 M", 915), ("1.83 m", 1830),
    ("915mm (36\")", None), ("2 x 915", None), ("", None), ("WIDE", None), ("N/A", None), ("-", None), ("1.2.3", None), ("0'-0\"", None), ("0.915", None), ("2", None), ("915 M", None), ("9" * 40, None),
    ("9" * 400 + "'-0\"", None), ("9" * 5000 + "'-0\"", None),
    ('69" CLEAR WIDTH', 1752.6), ('46" CLR.', 1168.4), ('CLEAR WIDTH: 36"', 914.4), ("3'-0\" CLEAR", 914.4), ("36 INCHES CLEAR WIDTH", 914.4), ("915 MM CLEAR", 915),
    ("CLEAR WIDTH", None), ("2 x 915 CLEAR", None), ("CLEARANCE 36", None),
])
def test_width_reads_to_mm(printed, want):
    assert width_mm(printed) == pytest.approx(want) if want else width_mm(printed) is None


def life_rows():
    return [read_row(0, "A18-1", "LIFE SAFETY FIRST FLOOR PLAN", "floor_plan")]


def test_life_safety_sheet_is_read_twice_in_four_tiles(tmp_path):
    ctx = review(tmp_path, life_rows())
    model = Model({"This image is part of a life safety": tile(), TB_START: {"sheet_number": "A18-1", "sheet_title": "LIFE SAFETY", "sequence": ""}, CODE_START: FIELDS})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count("This image is part of a life safety") == 8
    assert [s["sheet"] for s in out["exits"]["sheets"]] == [0]


def test_set_with_no_life_safety_sheet_says_so(tmp_path):
    ctx = review(tmp_path, [read_row(0, "A2.1", "FLOOR PLAN", "floor_plan")])
    meta.set_transport(Model({TB_START: {"sheet_number": "A2.1", "sheet_title": "FLOOR PLAN", "sequence": ""}}))
    out = declared_stage(ctx)
    assert out["exits"] == {"sheets": []}
    assert any(row["what"] == "Exit capacity" for row in out["cannot_tell"])


# ---- tables with no text layer ----

def ruled(rows=20, row_h=40, head_h=120, x0=300, y0=200, width=1600, canvas=(4000, 6000)):
    """A white sheet holding one ruled table with a tall header. Each body row carries a dark bar whose width grows with its number."""
    im = np.full((*canvas, 3), 255, np.uint8)
    top = y0 + head_h
    for k in range(rows + 1):
        cv2.line(im, (x0, top + k * row_h), (x0 + width, top + k * row_h), (0, 0, 0), 1)
    cv2.line(im, (x0, y0), (x0 + width, y0), (0, 0, 0), 1)
    for x in (x0, x0 + width // 2, x0 + width):
        cv2.line(im, (x, y0), (x, top + rows * row_h), (0, 0, 0), 1)
    for k in range(rows):
        cv2.rectangle(im, (x0 + 10, top + k * row_h + 12), (x0 + 20 + 5 * k, top + k * row_h + 26), (0, 0, 0), -1)
    cv2.rectangle(im, (x0 + 10, y0 + 40), (x0 + 700, y0 + 70), (0, 0, 0), -1)   # the header's own mark
    return im


def test_table_boxes_find_the_table_and_its_tall_header():
    box = table_boxes(ruled())[0]
    assert box == pytest.approx([300, 200, 1901, 1121], abs=3)


@pytest.mark.parametrize("x0, x1, gap", [(5300, 5950, 50), (300, 800, 50), (300, 1900, 100)], ids=["title-strip", "narrow-legend", "rules-too-far-apart"])
def test_table_boxes_ignore_rules_that_are_no_table(x0, x1, gap):
    im = np.full((4000, 6000, 3), 255, np.uint8)
    for k in range(10):
        cv2.line(im, (x0, 100 + gap * k), (x1, 100 + gap * k), (0, 0, 0), 1)
    assert table_boxes(im) == []


def test_table_boxes_put_the_largest_table_first():
    im = ruled(rows=20)
    other = ruled(rows=8, x0=3000, width=1200)
    im[:, 2900:] = other[:, 2900:]
    boxes = table_boxes(im)
    assert len(boxes) == 2 and boxes[0][0] == pytest.approx(300, abs=3)


def test_strips_cut_on_row_lines():
    rows, row_h, head_h, y0, x0, width = 60, 40, 120, 200, 300, 1600
    im = ruled(rows, row_h, head_h, x0, y0, width)
    box = [x0, y0, x0 + width + 1, y0 + head_h + rows * row_h + 1]
    strips, count = row_strips(im, box, per_strip=25, px=6000)
    assert count == rows
    assert len(strips) == 3   # 60 rows at 25 a strip
    seen = 0
    for image, (x1, y1, x2, y2) in strips:
        assert (y1 - (y0 + head_h)) % row_h == 0 and (y2 - y1) % row_h == 0   # both ends sit on row lines
        assert image[75:head_h - 2, 14].min() == 255   # the header stops at its rule, so no row sits under its own mark
        ink = (image[head_h + 3:, 14, 0] < 128).astype(np.uint8)
        runs = np.split(ink, np.flatnonzero(np.diff(ink)) + 1)
        bars = sum(len(run) >= 5 and run[0] == 1 for run in runs)
        assert bars == (y2 - y1) // row_h   # a whole bar for each row, and rules are too thin to count
        seen += bars
    assert seen == rows   # each row in one strip only


def test_short_table_is_one_strip_under_its_header():
    strips, count = row_strips(ruled(rows=6), [300, 200, 1901, 561], per_strip=25, px=6000)
    assert count == 6 and len(strips) == 1


def test_last_row_closed_by_the_sheet_border_stays():
    # USC A400: the final row has no rule across the table, so a cut at the last rule dropped door 317T1
    im = ruled(rows=8)
    bottom = 200 + 120 + 8 * 40
    cv2.rectangle(im, (310, bottom + 12), (360, bottom + 26), (0, 0, 0), -1)
    cv2.line(im, (300, bottom), (300, bottom + 40), (0, 0, 0), 1)
    cv2.line(im, (1000, bottom), (1000, bottom + 40), (0, 0, 0), 1)
    cv2.line(im, (0, bottom + 40), (5999, bottom + 40), (0, 0, 0), 1)   # the border runs past the table's ends
    box = table_boxes(im)[0]
    assert box[3] >= bottom + 40
    strips, count = row_strips(im, box, px=6000)
    assert count == 9
    assert strips[-1][0][-30:, 14, 0].min() == 0   # the last row's mark is in the strip


def test_table_with_uneven_rows_is_sent_whole():
    im = np.full((1000, 2000, 3), 255, np.uint8)
    for y in (100, 130, 200, 210, 400, 800):
        cv2.line(im, (100, y), (1900, y), (0, 0, 0), 1)
    strips, count = row_strips(im, [100, 100, 1900, 801], per_strip=25, px=2000)
    assert count == 0 and len(strips) == 1


def door_rows(*marks, width="915"):
    return {"rows": [{"door": m, "width": width, "fire_rating": "", "hardware": "101", "room": "OFFICE"} for m in marks]}


def schedule_ctx(tmp_path, title="DOOR SCHEDULE", cls="schedule"):
    ctx = review(tmp_path, [read_row(0, "A5-1", title, cls)])
    cv2.imwrite(str(tmp_path / "sheet_000.png"), ruled())
    return ctx


TB = {"sheet_number": "A5-1", "sheet_title": "DOOR SCHEDULE", "sequence": ""}


def test_schedule_with_no_text_is_read_twice_on_strips_and_keeps_rows_both_saw(tmp_path):
    ctx = schedule_ctx(tmp_path)
    model = Model({TB_START: TB, "This image contains a door schedule": [door_rows("1A03", "1A04", "1A05"), door_rows("1A03", "1A04")]})
    meta.set_transport(model)
    out = declared_stage(ctx)
    table = out["schedules"][0]
    assert (table["sheet"], table["kind"], table["source"]) == (0, "door", "spark")
    assert [r["mark"] for r in table["rows"]] == ["1A03", "1A04"]
    assert table["rows"][0]["cells"]["ROOM"] == "OFFICE"
    assert model.count("This image contains a door schedule") == 2
    assert any("1A05" in row["why"] for row in out["cannot_tell"])


def test_schedule_cell_two_reads_differ_on_is_left_empty(tmp_path):
    ctx = schedule_ctx(tmp_path)
    meta.set_transport(Model({TB_START: TB, "This image contains a door schedule": [door_rows("1A03", width="915"), door_rows("1A03", width="965")]}))
    out = declared_stage(ctx)
    assert out["schedules"][0]["rows"][0]["cells"]["WIDTH"] is None
    assert any("1A03" in row["why"] for row in out["cannot_tell"])


def test_finish_name_with_a_printed_typo_read_twice_stays(tmp_path):
    ctx = schedule_ctx(tmp_path, "FINISH SCHEDULE")
    finish = [{"rows": [{"room_no": "101", "room_name": "KTICHEN"}]}, {"rows": [{"room_no": "101", "room_name": "KITCHEN"}]}]
    meta.set_transport(Model({TB_START: TB, "This image contains a room finish schedule": finish}))
    assert declared_stage(ctx)["schedules"][0]["rows"][0]["cells"]["NAME"] == "KTICHEN"


def test_finish_name_one_read_left_blank_is_a_difference(tmp_path):
    ctx = schedule_ctx(tmp_path, "FINISH SCHEDULE")
    finish = [{"rows": [{"room_no": "101", "room_name": "KITCHEN"}]}, {"rows": [{"room_no": "101", "room_name": ""}]}]
    meta.set_transport(Model({TB_START: TB, FINISH_START: finish}))
    out = declared_stage(ctx)
    assert out["schedules"][0]["rows"][0]["cells"]["NAME"] is None
    assert any("101" in row["why"] for row in out["cannot_tell"])


def test_door_room_one_read_left_blank_is_a_difference(tmp_path):
    ctx = schedule_ctx(tmp_path)
    blank = {"rows": [{"door": "1A03", "width": "915", "fire_rating": "", "hardware": "101", "room": ""}]}
    meta.set_transport(Model({TB_START: TB, DOOR_START: [door_rows("1A03"), blank]}))
    out = declared_stage(ctx)
    assert out["schedules"][0]["rows"][0]["cells"]["ROOM"] is None
    assert any("1A03" in row["why"] for row in out["cannot_tell"])


def test_column_both_reads_leave_blank_stays_blank_and_makes_no_row(tmp_path):
    ctx = schedule_ctx(tmp_path)
    blank = {"rows": [{"door": "1A03", "width": "915", "fire_rating": "", "hardware": "101", "room": ""}]}
    meta.set_transport(Model({TB_START: TB, DOOR_START: blank}))
    out = declared_stage(ctx)
    assert out["schedules"][0]["rows"][0]["cells"]["ROOM"] == ""
    assert not any("1A03" in row["why"] for row in out["cannot_tell"])


def test_finish_names_that_differ_are_left_empty(tmp_path):
    ctx = schedule_ctx(tmp_path, "FINISH SCHEDULE")
    finish = [{"rows": [{"room_no": "101", "room_name": "OFFICE"}]}, {"rows": [{"room_no": "101", "room_name": "BOILER"}]}]
    meta.set_transport(Model({TB_START: TB, FINISH_START: finish}))
    assert declared_stage(ctx)["schedules"][0]["rows"][0]["cells"]["NAME"] is None


@pytest.mark.parametrize("reply", [BAD, {"rows": []}, [door_rows("1A03"), BAD], [door_rows("1A03"), {"rows": []}], [BAD, BAD, door_rows("1A03")], [{"rows": []}, door_rows("1A03")]],   # spark tries an empty reply twice
                         ids=["lost", "blank", "second-read-lost", "second-read-blank", "first-read-lost", "first-read-blank"])
def test_door_strip_lost_or_blank_is_cannot_tell(tmp_path, reply):
    ctx = schedule_ctx(tmp_path)
    meta.set_transport(Model({TB_START: TB, DOOR_START: reply}))
    out = declared_stage(ctx)
    assert out["schedules"] == []
    assert any(row["what"] == "Door schedule" and row["sheet"] == 0 and "1 of 1" in row["why"] for row in out["cannot_tell"])


def test_finish_strip_lost_is_cannot_tell(tmp_path):
    ctx = schedule_ctx(tmp_path, "FINISH SCHEDULE")
    meta.set_transport(Model({TB_START: TB, FINISH_START: BAD}))
    assert any(row["what"] == "Finish schedule" and "1 of 1" in row["why"] for row in declared_stage(ctx)["cannot_tell"])


def test_one_strip_of_three_lost_keeps_the_rest_and_says_so(tmp_path):
    ctx = schedule_ctx(tmp_path)
    cv2.imwrite(str(tmp_path / "sheet_000.png"), ruled(rows=60))
    meta.set_transport(Model({TB_START: TB, DOOR_START: by_crop(lambda n: BAD if n == 1 else door_rows(*[f"{n}A{k:02d}" for k in range(25)]))}))
    out = declared_stage(ctx)
    assert len(out["schedules"][0]["rows"]) == 50
    assert any(row["what"] == "Door schedule" and row["sheet"] == 0 and "1 of 3" in row["why"] for row in out["cannot_tell"])


def two_tables(tmp_path, *others, rows=20):
    """The sheet's image: ruled tables, largest first, the second and later ones further right."""
    im = ruled(rows=rows)
    for n, x in enumerate(others):
        im[:, x - 40:] = ruled(rows=20 - n, x0=x, width=900)[:, x - 40:]
    cv2.imwrite(str(tmp_path / "sheet_000.png"), im)


def test_sheet_with_two_tables_keeps_the_one_with_the_most_agreed_rows(tmp_path):
    ctx = schedule_ctx(tmp_path)
    two_tables(tmp_path, 3000)
    model = Model({TB_START: TB, DOOR_START: by_crop(lambda n: [door_rows("9A"), door_rows("1A03", "1A04", "1A05")][n])})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count(DOOR_START) == 4   # two reads of each table
    assert [r["mark"] for r in out["schedules"][0]["rows"]] == ["1A03", "1A04", "1A05"]
    assert out["schedules"][0]["rows"][0]["box"][0] >= 3000
    assert any(row["sheet"] == 0 and "2 ruled tables" in row["why"] for row in out["cannot_tell"])


def test_table_with_more_rows_but_no_width_does_not_stand_in_for_the_door_table(tmp_path):
    ctx = schedule_ctx(tmp_path)
    two_tables(tmp_path, 3000)
    meta.set_transport(Model({TB_START: TB, DOOR_START: by_crop(lambda n: [door_rows("H1", "H2", "H3", width=""), door_rows("1A03")][n])}))
    assert [r["mark"] for r in declared_stage(ctx)["schedules"][0]["rows"]] == ["1A03"]


def test_tables_that_tie_keep_the_largest(tmp_path):
    ctx = schedule_ctx(tmp_path)
    two_tables(tmp_path, 3000)
    meta.set_transport(Model({TB_START: TB, DOOR_START: by_crop(lambda n: [door_rows("1A03"), door_rows("9A")][n])}))
    assert [r["mark"] for r in declared_stage(ctx)["schedules"][0]["rows"]] == ["1A03"]


def test_rows_with_no_width_beat_a_table_read_with_nothing(tmp_path):
    ctx = schedule_ctx(tmp_path)
    two_tables(tmp_path, 3000)
    meta.set_transport(Model({TB_START: TB, DOOR_START: by_crop(lambda n: [{"rows": []}, door_rows("1A03", width="")][n])}))
    assert [r["mark"] for r in declared_stage(ctx)["schedules"][0]["rows"]] == ["1A03"]


def test_lost_table_beside_a_kept_one_is_cannot_tell(tmp_path):
    ctx = schedule_ctx(tmp_path)
    two_tables(tmp_path, 3000)
    meta.set_transport(Model({TB_START: TB, DOOR_START: by_crop(lambda n: [door_rows("H1", "H2"), BAD][n])}))
    out = declared_stage(ctx)
    assert [r["mark"] for r in out["schedules"][0]["rows"]] == ["H1", "H2"]
    assert any(row["what"] == "Door schedule" and row["sheet"] == 0 and "1 of 1" in row["why"] for row in out["cannot_tell"])


def test_table_not_kept_that_comes_back_blank_makes_no_row(tmp_path):
    # a hardware table under the door prompt: Spark rightly finds no doors there
    ctx = schedule_ctx(tmp_path)
    two_tables(tmp_path, 3000)
    meta.set_transport(Model({TB_START: TB, DOOR_START: by_crop(lambda n: [door_rows("1A03"), {"rows": []}][n])}))
    assert not any("crops" in row["why"] for row in declared_stage(ctx)["cannot_tell"])


def test_kept_table_with_a_blank_strip_is_cannot_tell(tmp_path):
    ctx = schedule_ctx(tmp_path)
    cv2.imwrite(str(tmp_path / "sheet_000.png"), ruled(rows=60))
    meta.set_transport(Model({TB_START: TB, DOOR_START: by_crop(lambda n: {"rows": []} if n == 1 else door_rows(*[f"{n}A{k:02d}" for k in range(25)]))}))
    assert any(row["what"] == "Door schedule" and "1 of 3" in row["why"] for row in declared_stage(ctx)["cannot_tell"])


def test_rows_only_one_read_saw_in_a_table_not_kept_make_no_row(tmp_path):
    ctx = schedule_ctx(tmp_path)
    two_tables(tmp_path, 3000)
    meta.set_transport(Model({TB_START: TB, DOOR_START: by_crop(lambda n: [door_rows("1A03", "1A04"), [door_rows("H1"), door_rows("H2")]][n])}))
    out = declared_stage(ctx)
    assert [r["mark"] for r in out["schedules"][0]["rows"]] == ["1A03", "1A04"]
    assert not any("H1" in row["why"] for row in out["cannot_tell"])


def test_strip_lost_from_the_kept_table_makes_one_row(tmp_path):
    ctx = schedule_ctx(tmp_path)
    two_tables(tmp_path, 3000, rows=60)
    door = by_crop(lambda n: {"rows": []} if n == 3 else BAD if n == 1 else door_rows(*[f"{n}A{k:02d}" for k in range(25)]))
    meta.set_transport(Model({TB_START: TB, DOOR_START: door}))
    assert [row["why"][:6] for row in declared_stage(ctx)["cannot_tell"] if "crops" in row["why"]] == ["1 of 3"]


def test_tables_past_the_cap_are_not_read(tmp_path):
    ctx = schedule_ctx(tmp_path)
    two_tables(tmp_path, 2000, 3000, 4000)
    model = Model({TB_START: TB, DOOR_START: door_rows("1A03")})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count(DOOR_START) == 2 * declared.MAX_TABLES
    assert any("4 ruled tables" in row["why"] and "largest 3" in row["why"] for row in out["cannot_tell"])


def test_door_and_finish_reads_on_their_own_tables_are_both_kept(tmp_path):
    ctx = schedule_ctx(tmp_path, "DOOR & FINISH SCHEDULES")
    two_tables(tmp_path, 3000)
    finish = {"rows": [{"room_no": "101", "room_name": "KITCHEN"}]}
    meta.set_transport(Model({TB_START: TB, DOOR_START: by_crop(lambda n: [door_rows("1A03", "1A04"), door_rows("9A")][n]), FINISH_START: by_crop(lambda n: [{"rows": []}, finish][n])}))
    assert [(t["kind"], t["rows"][0]["mark"]) for t in declared_stage(ctx)["schedules"]] == [("door", "1A03"), ("finish", "101")]


def test_door_and_finish_reads_on_different_sheets_are_both_kept(tmp_path):
    ctx = review(tmp_path, [read_row(0, "A5-1", "DOOR SCHEDULE", "schedule"), read_row(1, "A5-2", "FINISH SCHEDULE", "schedule")])
    for sheet in ("sheet_000.png", "sheet_001.png"):
        cv2.imwrite(str(tmp_path / sheet), ruled())
    meta.set_transport(Model({TB_START: TB, DOOR_START: door_rows("1A03"), FINISH_START: {"rows": [{"room_no": "101", "room_name": "KITCHEN"}]}}))
    assert [t["kind"] for t in declared_stage(ctx)["schedules"]] == ["door", "finish"]


def test_door_and_finish_reads_that_land_on_one_table_keep_neither(tmp_path):
    ctx = schedule_ctx(tmp_path, "DOOR & FINISH SCHEDULES")
    finish = {"rows": [{"room_no": "1A03", "room_name": "OFFICE"}]}
    meta.set_transport(Model({TB_START: TB, DOOR_START: door_rows("1A03"), FINISH_START: finish}))
    out = declared_stage(ctx)
    assert out["schedules"] == []
    assert {row["what"] for row in out["cannot_tell"] if row["sheet"] == 0 and "same table" in row["why"]} == {"Door schedule", "Finish schedule"}


def test_schedule_sheet_with_no_ruled_table_is_cannot_tell(tmp_path):
    ctx = review(tmp_path, [read_row(0, "A5-1", "DOOR SCHEDULE", "schedule")])
    model = Model({TB_START: TB})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert any(row["sheet"] == 0 and "table" in row["why"] for row in out["cannot_tell"])


def test_window_schedule_with_no_text_is_cannot_tell(tmp_path):
    ctx = schedule_ctx(tmp_path, "WINDOW SCHEDULE")
    meta.set_transport(Model({TB_START: TB}))
    out = declared_stage(ctx)
    assert out["schedules"] == []
    assert any(row["sheet"] == 0 and "window" in row["what"].lower() for row in out["cannot_tell"])


def test_no_text_code_sheet_reads_its_occupant_table(tmp_path):
    ctx = review(tmp_path, [read_row(0, "G002", "CODE SHEET", "code_summary")])
    cv2.imwrite(str(tmp_path / "sheet_000.png"), ruled())
    occ = {"rows": [{"area": "4,351GSF (Total)", "factor": "100/GSF", "occupants": "43"}]}
    meta.set_transport(Model({TB_START: TB, CODE_START: FIELDS, "This image is an occupant load table": occ}))
    tables = declared_stage(ctx)["tables"]
    assert [(t["sheet"], t["kind"]) for t in tables] == [(0, "occupant")]
    assert tables[0]["rows"][0] == {"area": 4351.0, "factor": 100.0, "occupants": 43}


def occupant(area="100", factor="10", occupants="10"):
    return {"area": area, "factor": factor, "occupants": occupants}


@pytest.mark.parametrize("reply, why, kept", [
    (BAD, "no row", 0),
    ({"rows": []}, "no row", 0),
    ([{"rows": [occupant()]}, BAD], "no row", 0),
    ([{"rows": [occupant()]}, {"rows": [occupant(occupants="11")]}], "no row", 0),
    ({"rows": [occupant(area="?")]}, "no row", 0),
    ([{"rows": [occupant(), occupant("200")]}, {"rows": [occupant()]}], "differ", 1),
    ([{"rows": [occupant()]}, {"rows": [occupant(), occupant("200")]}], "differ", 1),
], ids=["lost", "blank", "second-read-lost", "number-differs", "no-number", "first-read-adds-a-row", "second-read-adds-a-row"])
def test_occupant_crop_is_cannot_tell_unless_both_reads_give_the_same_rows(tmp_path, reply, why, kept):
    ctx = review(tmp_path, [read_row(0, "G002", "CODE SHEET", "code_summary")])
    cv2.imwrite(str(tmp_path / "sheet_000.png"), ruled())
    meta.set_transport(Model({TB_START: TB, CODE_START: FIELDS, OCC_START: reply}))
    out = declared_stage(ctx)
    assert sum(len(t["rows"]) for t in out["tables"]) == kept
    assert any(row["what"] == "Occupant table" and row["sheet"] == 0 and why in row["why"] for row in out["cannot_tell"])


def test_occupant_crop_both_reads_agree_on_makes_no_row(tmp_path):
    ctx = review(tmp_path, [read_row(0, "G002", "CODE SHEET", "code_summary")])
    cv2.imwrite(str(tmp_path / "sheet_000.png"), ruled())
    meta.set_transport(Model({TB_START: TB, CODE_START: FIELDS, OCC_START: {"rows": [occupant()]}}))
    assert not any(row["what"] == "Occupant table" for row in declared_stage(ctx)["cannot_tell"])


@pytest.mark.parametrize("cells", [(), ("101", "OFFICE", "109 SQ FT", "150", "1")], ids=["table-drawn-as-lines", "row-no-pattern-reads"])
def test_table_heading_with_no_rows_parsed_is_cannot_tell(tmp_path, cells):
    def draw(p, i):
        p.insert_text((2400, 1560), "GENERAL NOTES", fontsize=12.6)
        p.insert_text((2420, 1620), "A012", fontsize=31.5)
        p.insert_text((2027, 83), "OCCUPANCY CALCULATIONS", fontsize=25.2)
        for k, cell in enumerate(cells):
            p.insert_text((2000 + 80 * k, 314), cell, fontsize=9.5)
    ctx = pdf_review(tmp_path, draw, [read_row(0, "A012", "GENERAL NOTES", "other")])
    meta.set_transport(Model({}))
    assert any(row["sheet"] == 0 and "occupant" in row["what"].lower() for row in declared_stage(ctx)["cannot_tell"])


# ---- sheet numbers and titles with no text layer ----

def index_rows(*rows):
    return {"rows": [{"sheet_number": n, "sequence": "", "title": t} for n, t in rows]}


def titled_ctx(tmp_path, rows=None):
    return review(tmp_path, rows or [read_row(0, "T1", "INDEX OF DRAWINGS", "other"), read_row(1, "A2-1", "FIRST FLOOR PLAN", "floor_plan")])


def block_read(number, title):
    return {"sheet_number": number, "sheet_title": title, "sequence": ""}


def test_spark_index_is_read_in_tiles_twice_and_keeps_rows_both_agree_on(tmp_path):
    ctx = titled_ctx(tmp_path)
    tb = by_sheet({0: block_read("T1", "INDEX OF DRAWINGS"), 1: block_read("A2-1", "FIRST FLOOR PLAN")})
    good = index_rows(("T1", "INDEX OF DRAWINGS"), ("A2-1", "FIRST FLOOR PLAN"), ("A3-1", "(NOT USED)"))
    other = index_rows(("T1", "INDEX OF DRAWINGS"), ("A2-1", "FIRST FLOOR PLAN"), ("A3-1", "(NOT USED)"), ("A4-1", "SECTIONS"))
    model = Model({TB_START: tb, INDEX_START: [good, other]})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count(INDEX_START) == 16
    assert [(r["number"], r["not_used"], r["sheet"]) for r in out["index"]] == [("T1", False, 0), ("A2-1", False, 0), ("A3-1", True, 0)]
    assert any("A4-1" in row["why"] for row in out["cannot_tell"])


def index_titles():
    return by_sheet({0: block_read("T1", "INDEX OF DRAWINGS"), 1: block_read("A2-1", "FIRST FLOOR PLAN")})


@pytest.mark.parametrize("lost", [BAD, [index_rows(("T1", "INDEX OF DRAWINGS")), BAD]], ids=["both-reads", "second-read"])
def test_index_tile_lost_is_cannot_tell(tmp_path, lost):
    ctx = titled_ctx(tmp_path)
    good = index_rows(("T1", "INDEX OF DRAWINGS"))
    meta.set_transport(Model({TB_START: index_titles(), INDEX_START: by_crop(lambda n: lost if n == 0 else good)}))
    out = declared_stage(ctx)
    assert [r["number"] for r in out["index"]] == ["T1"]
    assert any(row["what"] == "Sheet index" and row["sheet"] == 0 and "1 of 8" in row["why"] for row in out["cannot_tell"])


def test_index_tile_of_the_wrong_shape_is_lost(tmp_path):
    ctx = titled_ctx(tmp_path)
    good = index_rows(("T1", "INDEX OF DRAWINGS"))
    meta.set_transport(Model({TB_START: index_titles(), INDEX_START: by_crop(lambda n: {"rows": [{"sheet_number": 7}]} if n == 0 else good)}))
    assert any(row["sheet"] == 0 and "1 of 8" in row["why"] for row in declared_stage(ctx)["cannot_tell"])


def test_index_tile_with_no_rows_is_not_lost(tmp_path):
    ctx = titled_ctx(tmp_path)
    good = index_rows(("T1", "INDEX OF DRAWINGS"))
    meta.set_transport(Model({TB_START: index_titles(), INDEX_START: by_crop(lambda n: {"rows": []} if n == 0 else good)}))
    assert not any(row["what"] == "Sheet index" for row in declared_stage(ctx)["cannot_tell"])


def test_index_titles_two_reads_differ_on_are_cannot_tell(tmp_path):
    ctx = titled_ctx(tmp_path)
    meta.set_transport(Model({TB_START: index_titles(), INDEX_START: [index_rows(("A2-1", "FIRST FLOOR PLAN")), index_rows(("A2-1", "ROOF DRAINAGE DETAILS"))]}))
    out = declared_stage(ctx)
    assert out["index"] == []
    assert any(row["what"] == "Sheet index" and "A2-1" in row["why"] for row in out["cannot_tell"])


def test_second_title_read_runs_only_where_the_first_disagrees_with_the_index(tmp_path):
    ctx = titled_ctx(tmp_path)
    tb = by_sheet({0: block_read("T1", "INDEX OF DRAWINGS"), 1: block_read("A2-1", "SECOND FLOOR PLAN")})
    index = index_rows(("T1", "INDEX OF DRAWINGS"), ("A2-1", "FIRST FLOOR PLAN"))
    model = Model({TB_START: tb, INDEX_START: index, "Two strings": {"results": [{"id": "1", "answer": "different"}]}})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count(TB_START) == 3   # two sheets, then a second read of the one that disagrees
    assert out["titles"][1]["judge"] == "different"
    assert out["titles"][0]["judge"] is None


def test_two_title_reads_disagree_on_the_number_is_cannot_tell(tmp_path):
    ctx = titled_ctx(tmp_path)
    tb = by_sheet({0: block_read("T1", "INDEX OF DRAWINGS"), 1: [block_read("A2-7", "FIRST FLOOR PLAN"), block_read("A2-9", "FIRST FLOOR PLAN")]})
    meta.set_transport(Model({TB_START: tb, INDEX_START: index_rows(("T1", "INDEX OF DRAWINGS"))}))
    out = declared_stage(ctx)
    assert out["titles"][1]["number"] == ""
    assert [row["what"] for row in out["cannot_tell"] if row["sheet"] == 1 and row["what"].startswith("Sheet")] == ["Sheet number"]
    assert any("A2-7" in row["why"] and "A2-9" in row["why"] for row in out["cannot_tell"])


def test_sheet_read_that_matches_the_index_saves_a_second_read(tmp_path):
    ctx = titled_ctx(tmp_path)
    tb = by_sheet({0: block_read("T1", "INDEX OF DRAWINGS"), 1: block_read("A2-1", "DRAWN BY")})   # the corner holds the number and not the title
    model = Model({TB_START: tb, INDEX_START: index_rows(("T1", "INDEX OF DRAWINGS"), ("A2-1", "FIRST FLOOR PLAN"))})
    meta.set_transport(model)
    assert declared_stage(ctx)["titles"][1]["title"] == "FIRST FLOOR PLAN"
    assert model.count(TB_START) == 2


def test_second_read_lost_leaves_the_sheet_unread(tmp_path):
    ctx = titled_ctx(tmp_path)
    tb = by_sheet({0: block_read("T1", "INDEX OF DRAWINGS"), 1: [block_read("A2-1", "SECOND FLOOR PLAN"), {"text": None}]})
    meta.set_transport(Model({TB_START: tb, INDEX_START: index_rows(("T1", "INDEX OF DRAWINGS"), ("A2-1", "FIRST FLOOR PLAN"))}))
    out = declared_stage(ctx)
    assert out["titles"][1]["number"] == "" and any(row["sheet"] == 1 for row in out["cannot_tell"])


def test_two_title_reads_that_differ_get_no_judge(tmp_path):
    ctx = titled_ctx(tmp_path)
    tb = by_sheet({0: block_read("T1", "INDEX OF DRAWINGS"), 1: [block_read("A2-1", "SECOND FLOOR PLAN"), block_read("A2-1", "PLUMBING DETAILS")]})
    model = Model({TB_START: tb, INDEX_START: index_rows(("T1", "INDEX OF DRAWINGS"), ("A2-1", "FIRST FLOOR PLAN"))})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count("Two strings") == 0 and out["titles"][1]["judge"] is None
    assert any(row["sheet"] == 1 and "PLUMBING DETAILS" in row["why"] for row in out["cannot_tell"])


def test_blank_title_crop_falls_back_on_the_sheet_read(tmp_path):
    ctx = titled_ctx(tmp_path)
    meta.set_transport(Model({TB_START: block_read("", "")}))
    out = declared_stage(ctx)
    assert (out["titles"][1]["number"], out["titles"][1]["title"]) == ("A2-1", "FIRST FLOOR PLAN")


def test_title_read_that_fails_falls_back_and_says_nothing_false(tmp_path):
    ctx = titled_ctx(tmp_path)
    meta.set_transport(Model({TB_START: {"text": None}}))
    assert declared_stage(ctx)["titles"][1]["number"] == "A2-1"


def test_sheet_with_a_text_layer_no_number_and_every_read_lost_is_cannot_tell(tmp_path):
    ctx = pdf_review(tmp_path, lambda p, i: p.insert_text((100, 100), "GENERAL NOTES ONLY HERE", fontsize=12.6), [read_row(0, read_ok=False)])
    meta.set_transport(Model({TB_START: BAD}))
    out = declared_stage(ctx)
    assert (out["titles"][0]["number"], out["titles"][0]["title"]) == ("", "")
    assert any(row["what"] == "Sheet title" and row["sheet"] == 0 for row in out["cannot_tell"])


def test_sheet_with_a_title_and_no_number_is_cannot_tell(tmp_path):
    ctx = titled_ctx(tmp_path, [read_row(0, "", "FIRST FLOOR PLAN", "floor_plan")])
    meta.set_transport(Model({TB_START: block_read("", "")}))
    out = declared_stage(ctx)
    assert (out["titles"][0]["number"], out["titles"][0]["title"]) == ("", "FIRST FLOOR PLAN")
    assert any(row["what"] == "Sheet title" and row["sheet"] == 0 for row in out["cannot_tell"])


def test_text_sheet_with_a_number_and_no_title_is_cannot_tell(tmp_path):
    ctx = pdf_review(tmp_path, lambda p, i: p.insert_text((2420, 1620), "A1.0", fontsize=31.5), [read_row(0, "A1.0")])
    meta.set_transport(Model({}))
    out = declared_stage(ctx)
    assert (out["titles"][0]["number"], out["titles"][0]["title"]) == ("A1.0", "")
    assert any(row["what"] == "Sheet title" and row["sheet"] == 0 for row in out["cannot_tell"])


def test_sheet_with_a_number_and_a_title_makes_no_row(tmp_path):
    ctx = titled_ctx(tmp_path)
    meta.set_transport(Model({TB_START: BAD}))
    assert not any(row["what"] == "Sheet title" for row in declared_stage(ctx)["cannot_tell"])


@pytest.mark.parametrize("first, second, want", [("different", "different", "different"), ("same", "different", "same"), ("same", "same", "same")])
def test_judge_says_different_only_when_both_reads_do(tmp_path, first, second, want):
    def draw(p, i):
        p.insert_text((2400, 1560), ["COVER", "HVAC COVER SHEET"][i], fontsize=12.6)
        p.insert_text((2420, 1620), ["A1.0", "M0.1"][i], fontsize=31.5)
        if i == 0:
            for n, row in enumerate(("A1.0  Cover", "M0.1  Plumbing Schedules", "A3.0  Sections", "A4.0  Elevations")):
                p.insert_text((100, 200 + 22 * n), row, fontsize=18.9)
    ctx = pdf_review(tmp_path, draw, [read_row(0, "A1.0", "COVER", "other"), read_row(1, "M0.1", "HVAC COVER SHEET", "other")], pages=2)
    results = [{"results": [{"id": "1", "answer": a}]} for a in (first, second)]
    meta.set_transport(Model({"Two strings": results}))
    assert declared_stage(ctx)["titles"][1]["judge"] == want


def test_number_listed_twice_on_the_index_matches_either_row(tmp_path):
    # Baxter lists E2.1 as Floor Plans - Power and as Electrical Details, and the sheet is the first of them
    def draw(p, i):
        p.insert_text((2400, 1560), ["COVER", "FLOOR PLANS - POWER"][i], fontsize=12.6)
        p.insert_text((2420, 1620), ["A1.0", "E2.1"][i], fontsize=31.5)
        if i == 0:
            for n, row in enumerate(("A1.0  Cover", "E2.1  Floor Plans - Power", "E2.1  Electrical Details", "A4.0  Elevations")):
                p.insert_text((100, 200 + 22 * n), row, fontsize=18.9)
    ctx = pdf_review(tmp_path, draw, [read_row(0, "A1.0", "COVER", "other"), read_row(1, "E2.1", "FLOOR PLANS - POWER", "other")], pages=2)
    model = Model({})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count("Two strings") == 0
    assert [r["number"] for r in out["index"]].count("E2.1") == 2   # the repeat stays for the index check to see


def test_judge_that_fails_leaves_no_verdict(tmp_path):
    def draw(p, i):
        p.insert_text((2400, 1560), ["COVER", "HVAC COVER SHEET"][i], fontsize=12.6)
        p.insert_text((2420, 1620), ["A1.0", "M0.1"][i], fontsize=31.5)
        if i == 0:
            for n, row in enumerate(("A1.0  Cover", "M0.1  Plumbing Schedules", "A3.0  Sections", "A4.0  Elevations")):
                p.insert_text((100, 200 + 22 * n), row, fontsize=18.9)
    ctx = pdf_review(tmp_path, draw, [read_row(0, "A1.0", "COVER", "other"), read_row(1, "M0.1", "HVAC COVER SHEET", "other")], pages=2)
    meta.set_transport(Model({"Two strings": {"text": "not json"}}))
    out = declared_stage(ctx)
    assert out["titles"][1]["judge"] is None
    assert any("M0.1" in row["what"] + row["why"] for row in out["cannot_tell"])


def test_floor_words_in_conflict_on_a_scan_get_two_label_reads(tmp_path):
    ctx = titled_ctx(tmp_path, [read_row(0, "A2.1", "MAIN FLOOR PLAN", "floor_plan", "GROUND FLOOR")])
    label = {"sheet_number": "A2.1", "sheet_title": "MAIN FLOOR PLAN", "drawing_title": "GROUND FLOOR PLAN"}
    model = Model({TB_START: block_read("A2.1", "MAIN FLOOR PLAN"), LABEL_START: label})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count(LABEL_START) == 2
    assert out["titles"][0]["under_drawing"] == "GROUND FLOOR PLAN"


@pytest.mark.parametrize("titles", [("FIRST FLOOR PLAN", "FIRST FLOOR PLAN"), ("SECOND FLOOR PLAN", "FIRST FLOOR PLAN"), ("FIRST FLOOR PLAN", "SECOND FLOOR PLAN"), ("SECOND FLOOR PLAN", "PLAN")],
                         ids=["both-reads", "second-read", "first-read", "no-floor-word"])
def test_title_crop_the_whole_sheet_reads_do_not_back_leaves_no_conflict(tmp_path, titles):
    ctx = titled_ctx(tmp_path, [read_row(0, "A2.1", "FIRST FLOOR PLAN", "floor_plan", "FIRST FLOOR")])
    labels = [{"sheet_number": "A2.1", "sheet_title": t, "drawing_title": "FIRST FLOOR PLAN"} for t in titles]
    meta.set_transport(Model({TB_START: block_read("A2.1", "SECOND FLOOR PLAN"), LABEL_START: labels}))
    out = declared_stage(ctx)
    assert out["titles"][0]["under_drawing"] is None
    assert any(row["what"] == "Sheet title" and row["sheet"] == 0 and "SECOND FLOOR PLAN" in row["why"] for row in out["cannot_tell"])


def test_title_crop_both_whole_sheet_reads_back_keeps_the_conflict(tmp_path):
    ctx = titled_ctx(tmp_path, [read_row(0, "A2.1", "SECOND FLOOR PLAN", "floor_plan", "FIRST FLOOR")])
    label = {"sheet_number": "A2.1", "sheet_title": "2ND FLOOR PLAN", "drawing_title": "FIRST FLOOR PLAN"}
    meta.set_transport(Model({TB_START: block_read("A2.1", "SECOND FLOOR PLAN"), LABEL_START: label}))
    out = declared_stage(ctx)
    assert (out["titles"][0]["title"], out["titles"][0]["under_drawing"]) == ("SECOND FLOOR PLAN", "FIRST FLOOR PLAN")
    assert not any(row["what"] == "Sheet title" for row in out["cannot_tell"])


def test_title_the_whole_sheet_reads_do_not_back_gets_no_judge(tmp_path):
    ctx = titled_ctx(tmp_path, [read_row(0, "T1", "INDEX OF DRAWINGS", "other"), read_row(1, "A2-1", "FIRST FLOOR PLAN", "floor_plan", "FIRST FLOOR")])
    index = index_rows(("T1", "INDEX OF DRAWINGS"), ("A2-1", "FIRST FLOOR PLAN"))
    label = {"sheet_number": "A2-1", "sheet_title": "FIRST FLOOR PLAN", "drawing_title": "FIRST FLOOR PLAN"}
    tb = by_sheet({0: block_read("T1", "INDEX OF DRAWINGS"), 1: block_read("A2-1", "SECOND FLOOR PLAN")})
    model = Model({TB_START: tb, INDEX_START: index, LABEL_START: label})
    meta.set_transport(model)
    assert declared_stage(ctx)["titles"][1]["judge"] is None
    assert model.count("Two strings") == 0


def test_label_reads_that_differ_leave_no_label(tmp_path):
    ctx = titled_ctx(tmp_path, [read_row(0, "A2.1", "MAIN FLOOR PLAN", "floor_plan", "GROUND FLOOR")])
    labels = [{"sheet_number": "A2.1", "sheet_title": "MAIN FLOOR PLAN", "drawing_title": t} for t in ("GROUND FLOOR PLAN", "UPPER FLOOR PLAN")]
    meta.set_transport(Model({TB_START: block_read("A2.1", "MAIN FLOOR PLAN"), LABEL_START: labels}))
    out = declared_stage(ctx)
    assert out["titles"][0]["under_drawing"] is None
    assert any(row["sheet"] == 0 for row in out["cannot_tell"])


# ---- labels and tags with no text layer ----

ROOMS = "This image is part of a floor plan. Room labels"
TAGS = "This image is part of a floor plan. Door tags"


def plans_ctx(tmp_path, schedule="FINISH SCHEDULE"):
    rows = [read_row(0, "A2-1", "FIRST FLOOR PLAN", "floor_plan", "FIRST FLOOR", [400, 400, 4000, 3000]), read_row(1, "A5-1", schedule, "schedule")]
    ctx = review(tmp_path, rows)
    table = ruled()
    table[-200:, -200:] = 1   # sheet_of reads this corner
    cv2.imwrite(str(tmp_path / "sheet_001.png"), table)
    write_json(tmp_path / "stage_floors.json", {"order": [0]})
    return ctx


def test_plan_labels_are_read_in_nine_tiles_twice(tmp_path):
    ctx = plans_ctx(tmp_path)
    tb = by_sheet({0: block_read("A2-1", "FIRST FLOOR PLAN"), 1: block_read("A5-1", "FINISH SCHEDULE")})
    finish = {"rows": [{"room_no": "1E08", "room_name": "OFFICE"}]}
    reads = [{"rooms": [{"number": "1e 08", "name": "OFFICE", "x": 500, "y": 500}, {"number": "1E09", "name": "STORAGE", "x": 100, "y": 100}]}, {"rooms": [{"number": "1E08", "name": "OFFICE", "x": 505, "y": 500}]}]
    model = Model({TB_START: tb, "This image contains a room finish schedule": finish, ROOMS: reads})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count(ROOMS) == 18
    assert [(r["sheet"], r["number"], r["name"], r["source"]) for r in out["rooms"]] == [(0, "1E08", "OFFICE", "spark")]
    assert out["rooms"][0]["box"][2] > out["rooms"][0]["box"][0]


def test_labels_are_not_read_when_no_schedule_needs_them(tmp_path):
    rows = [read_row(0, "A2-1", "FIRST FLOOR PLAN", "floor_plan", "FIRST FLOOR")]
    ctx = review(tmp_path, rows)
    write_json(tmp_path / "stage_floors.json", {"order": [0]})
    model = Model({TB_START: block_read("A2-1", "FIRST FLOOR PLAN")})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count(ROOMS) == 0 and model.count(TAGS) == 0
    assert not any(row["what"] == "Room labels" for row in out["cannot_tell"])


def test_door_tags_with_no_text_are_read_in_four_tiles_twice_and_both_must_see_them(tmp_path):
    ctx = plans_ctx(tmp_path, "DOOR SCHEDULE")
    tb = by_sheet({0: block_read("A2-1", "FIRST FLOOR PLAN"), 1: block_read("A5-1", "DOOR SCHEDULE")})
    reads = [{"tags": [{"tag": "317A1", "x": 500, "y": 500}, {"tag": "317B1", "x": 100, "y": 100}]}, {"tags": [{"tag": "317A1", "x": 500, "y": 505}]}]
    model = Model({TB_START: tb, "This image contains a door schedule": door_rows("317A1"), TAGS: reads})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.count(TAGS) == 8
    assert [(t["sheet"], t["kind"], t["mark"]) for t in out["tags"]] == [(0, "door", "317A1")]
    assert any("317B1" in row["why"] for row in out["cannot_tell"])


def plan_titles(schedule):
    return by_sheet({0: block_read("A2-1", "FIRST FLOOR PLAN"), 1: block_read("A5-1", schedule)})


FINISH_ROW = {"rows": [{"room_no": "1E08", "room_name": "OFFICE"}]}


@pytest.mark.parametrize("reply, why", [(BAD, "4 of 4"), ({"tags": []}, "No entry")], ids=["lost", "blank"])
def test_plan_with_no_door_tag_read_is_cannot_tell(tmp_path, reply, why):
    ctx = plans_ctx(tmp_path, "DOOR SCHEDULE")
    meta.set_transport(Model({TB_START: plan_titles("DOOR SCHEDULE"), DOOR_START: door_rows("317A1"), TAGS: reply}))
    out = declared_stage(ctx)
    assert out["tags"] == []
    assert any(row["what"] == "Door tags" and row["sheet"] == 0 and why in row["why"] for row in out["cannot_tell"])


def test_one_tag_tile_lost_keeps_the_rest_and_says_so(tmp_path):
    ctx = plans_ctx(tmp_path, "DOOR SCHEDULE")
    good = {"tags": [{"tag": "317A1", "x": 500, "y": 500}]}
    meta.set_transport(Model({TB_START: plan_titles("DOOR SCHEDULE"), DOOR_START: door_rows("317A1"), TAGS: by_crop(lambda n: BAD if n == 0 else good)}))
    out = declared_stage(ctx)
    assert [t["mark"] for t in out["tags"]] == ["317A1"]
    assert any(row["what"] == "Door tags" and row["sheet"] == 0 and "1 of 4" in row["why"] for row in out["cannot_tell"])


def test_tag_on_a_grid_past_the_tile_is_left_out(tmp_path):
    ctx = plans_ctx(tmp_path, "DOOR SCHEDULE")
    meta.set_transport(Model({TB_START: plan_titles("DOOR SCHEDULE"), DOOR_START: door_rows("317A1"), TAGS: {"tags": [{"tag": "317A1", "x": 1500, "y": 500}]}}))
    assert declared_stage(ctx)["tags"] == []


def test_tag_tile_with_no_tags_beside_a_plan_with_tags_makes_no_row(tmp_path):
    ctx = plans_ctx(tmp_path, "DOOR SCHEDULE")
    good = {"tags": [{"tag": "317A1", "x": 500, "y": 500}]}
    meta.set_transport(Model({TB_START: plan_titles("DOOR SCHEDULE"), DOOR_START: door_rows("317A1"), TAGS: by_crop(lambda n: {"tags": []} if n == 0 else good)}))
    assert not any(row["what"] == "Door tags" for row in declared_stage(ctx)["cannot_tell"])


@pytest.mark.parametrize("reply, why", [(BAD, "9 of 9"), ({"rooms": []}, "No entry")], ids=["lost", "blank"])
def test_plan_with_no_room_label_read_is_cannot_tell(tmp_path, reply, why):
    ctx = plans_ctx(tmp_path)
    meta.set_transport(Model({TB_START: plan_titles("FINISH SCHEDULE"), FINISH_START: FINISH_ROW, ROOMS: reply}))
    out = declared_stage(ctx)
    assert out["rooms"] == []
    assert any(row["what"] == "Room labels" and row["sheet"] == 0 and why in row["why"] for row in out["cannot_tell"])


def test_one_label_tile_lost_keeps_the_rest_and_says_so(tmp_path):
    ctx = plans_ctx(tmp_path)
    good = {"rooms": [{"number": "1E08", "name": "OFFICE", "x": 500, "y": 500}]}
    meta.set_transport(Model({TB_START: plan_titles("FINISH SCHEDULE"), FINISH_START: FINISH_ROW, ROOMS: by_crop(lambda n: BAD if n == 0 else good)}))
    out = declared_stage(ctx)
    assert [r["number"] for r in out["rooms"]] == ["1E08"]
    assert any(row["what"] == "Room labels" and row["sheet"] == 0 and "1 of 9" in row["why"] for row in out["cannot_tell"])


@pytest.mark.parametrize("name", ["BOILER", ""], ids=["differs", "second-read-blank"])
def test_room_label_two_reads_do_not_share_a_name_for_is_cannot_tell(tmp_path, name):
    ctx = plans_ctx(tmp_path)
    reads = [{"rooms": [{"number": "1E08", "name": "OFFICE", "x": 500, "y": 500}]}, {"rooms": [{"number": "1E08", "name": name, "x": 500, "y": 500}]}]
    meta.set_transport(Model({TB_START: plan_titles("FINISH SCHEDULE"), FINISH_START: FINISH_ROW, ROOMS: reads}))
    out = declared_stage(ctx)
    assert out["rooms"] == []
    assert any(row["what"] == "Room labels" and "1E08" in row["why"] for row in out["cannot_tell"])


def test_room_label_seen_by_one_read_is_left_out(tmp_path):
    ctx = plans_ctx(tmp_path)
    reads = [{"rooms": [{"number": "1E09", "name": "STORAGE", "x": 100, "y": 100}]}, {"rooms": []}]
    meta.set_transport(Model({TB_START: plan_titles("FINISH SCHEDULE"), FINISH_START: FINISH_ROW, ROOMS: reads}))
    assert declared_stage(ctx)["rooms"] == []


@pytest.mark.parametrize("heading, kind", [("WINDOW SCHEDULE", "Window"), ("DOOR SCHEDULE", "Door")])
def test_text_schedule_with_no_plan_to_compare_is_cannot_tell(tmp_path, heading, kind):
    # one sheet and no storey, so the door path has no plan for Spark to read either
    ctx = pdf_review(tmp_path, schedule_and_plans(heading), schedule_rows(1))
    meta.set_transport(Model({}))
    out = declared_stage(ctx)
    assert (out["schedules"][0]["kind"], out["schedules"][0]["plans"]) == (kind.lower(), [])
    assert any(row["what"] == f"{kind} schedule" and row["sheet"] == 0 and "plan" in row["why"] for row in out["cannot_tell"])


def test_window_schedule_on_a_set_with_plans_but_no_window_tags_is_cannot_tell(tmp_path):
    ctx = pdf_review(tmp_path, schedule_and_plans("WINDOW SCHEDULE"), schedule_rows(2), pages=2)
    meta.set_transport(Model({}))
    out = declared_stage(ctx)
    assert out["schedules"][0]["plans"] == []
    assert any(row["what"] == "Window schedule" and row["sheet"] == 0 for row in out["cannot_tell"])


def test_spark_door_schedule_with_no_plan_to_compare_is_cannot_tell(tmp_path):
    ctx = schedule_ctx(tmp_path)
    meta.set_transport(Model({TB_START: TB, DOOR_START: door_rows("1A03")}))
    out = declared_stage(ctx)
    assert out["schedules"][0]["plans"] == []
    assert any(row["what"] == "Door schedule" and "plan" in row["why"] for row in out["cannot_tell"])


def test_paired_schedule_makes_no_row_about_plans(tmp_path):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    meta.set_transport(Model({CODE_START: FIELDS}))
    assert not any("schedule" in row["what"] for row in declared_stage(ctx)["cannot_tell"])


def test_finish_schedule_with_no_label_read_on_any_plan_is_cannot_tell(tmp_path):
    ctx = pdf_review(tmp_path, schedule_and_plans("FINISH SCHEDULE"), schedule_rows(1))
    meta.set_transport(Model({}))
    out = declared_stage(ctx)
    assert out["rooms"] == [] and out["schedules"][0]["kind"] == "finish"
    assert any(row["what"] == "Room labels" for row in out["cannot_tell"])


def test_schedule_with_labels_read_makes_no_row_about_labels(tmp_path):
    ctx = plans_ctx(tmp_path)
    good = {"rooms": [{"number": "1E08", "name": "OFFICE", "x": 500, "y": 500}]}
    meta.set_transport(Model({TB_START: plan_titles("FINISH SCHEDULE"), FINISH_START: FINISH_ROW, ROOMS: good}))
    assert not any(row["what"] in ("Room labels", "Finish schedule") for row in declared_stage(ctx)["cannot_tell"])


# ---- spending and failures ----

def test_each_call_keeps_its_token_cap_and_second_reads_are_real_calls(tmp_path):
    ctx = review(tmp_path, [read_row(0, "G002", "CODE SHEET", "code_summary")])
    seen = []

    def transport(kind, **request):
        seen.append(request)
        return text(FIELDS if prompt_of(request).startswith(CODE_START) else block_read("G002", "CODE SHEET"))
    meta.set_transport(transport)
    declared_stage(ctx)
    assert sum(prompt_of(r).startswith(CODE_START) for r in seen) == 2
    assert all(r["max_completion_tokens"] > 0 for r in seen)
    assert all("run" not in r for r in seen)


def test_budget_stop_keeps_the_text_reads_and_says_so(tmp_path):
    ctx = pdf_review(tmp_path, text_set, text_rows(), pages=2)
    ctx.meter.cap_usd = 0
    model = Model({})
    meta.set_transport(model)
    out = declared_stage(ctx)
    assert model.calls == []
    assert len(out["index"]) == 4 and out["schedules"]
    assert any("spend cap" in row["why"] for row in out["cannot_tell"])


def test_api_error_that_is_not_about_one_sheet_fails_the_stage(tmp_path, monkeypatch):
    ctx = review(tmp_path, [read_row(0, "A2.1", "FLOOR PLAN", "floor_plan")])
    monkeypatch.setattr(meta.time, "sleep", lambda s: None)

    def transport(kind, **request):
        raise openai.APIStatusError("no", response=httpx.Response(500, request=httpx.Request("POST", "http://x")), body=None)
    meta.set_transport(transport)
    with pytest.raises(openai.APIStatusError):
        declared_stage(ctx)


def test_refused_image_is_lost_for_that_read_only(tmp_path):
    ctx = titled_ctx(tmp_path, [read_row(0, "G002", "CODE SHEET", "code_summary")])
    refused = openai.BadRequestError("no", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)
    meta.set_transport(Model({TB_START: block_read("G002", "CODE SHEET"), CODE_START: refused}))
    out = declared_stage(ctx)
    assert out["code"]["area"]["value"] is None
    assert any(row["sheet"] == 0 for row in out["cannot_tell"])


def test_rerun_comes_from_the_cache(tmp_path):
    ctx = review(tmp_path, [read_row(0, "G002", "CODE SHEET", "code_summary")])
    model = Model({TB_START: block_read("G002", "CODE SHEET"), CODE_START: FIELDS})
    meta.set_transport(model)
    declared_stage(ctx)
    calls = len(model.calls)
    spent = ctx.meter.spent
    declared_stage(ctx)
    assert len(model.calls) == calls and ctx.meter.spent == spent
    ctx.fresh = True
    declared_stage(ctx)
    assert len(model.calls) == 2 * calls


def test_progress_reaches_the_status_file(tmp_path):
    ctx = review(tmp_path, [read_row(0, "G002", "CODE SHEET", "code_summary")])
    ctx.status = {"stages": [{"state": "running"}]}
    meta.set_transport(Model({TB_START: block_read("G002", "CODE SHEET"), CODE_START: FIELDS}))
    declared_stage(ctx)
    assert ctx.status["step"] and ctx.status["stages"][0]["total"] > 0


def test_plan_box_too_small_to_crop_is_a_lost_read_not_a_crash(tmp_path):
    ctx = plans_ctx(tmp_path, "DOOR SCHEDULE")
    rows = json.loads((tmp_path / "stage_read.json").read_text())["sheets"]
    rows[0]["plan_box"] = [10, 10, 10, 10]
    write_json(tmp_path / "stage_read.json", {"sheets": rows})
    meta.set_transport(Model({TB_START: plan_titles("DOOR SCHEDULE"), DOOR_START: door_rows("317A1")}))
    assert any(row["what"] == "Door tags" and "4 of 4" in row["why"] for row in declared_stage(ctx)["cannot_tell"])


def test_unreadable_sheet_image_is_a_lost_read_not_a_crash(tmp_path):
    ctx = review(tmp_path, [read_row(0, "G002", "CODE SHEET", "code_summary")])
    (tmp_path / "sheet_000.png").write_bytes(b"not a png")
    meta.set_transport(Model({}))
    out = declared_stage(ctx)
    assert any(row["sheet"] == 0 for row in out["cannot_tell"])


# ---- the sheet selector ----

def rows_by_index(*rows):
    return {r["index"]: r for r in rows}


def picks(rows, pages=None, floors=None):
    rows = rows_by_index(*rows)
    return select(sorted(rows), rows, text_reads(pages or {}), floors or {})


def test_code_sheet_by_class():
    assert picks([read_row(0, "A1", "COVER", "cover"), read_row(1, "G2", "GENERAL DATA", "code_summary")])["code"] == [1]


def test_code_sheet_by_title_beats_the_cover():
    assert picks([read_row(0, "A1", "COVER", "cover"), read_row(1, "G2", "CODE SHEET, GENERAL NOTES", "other")])["code"] == [1]


def test_cover_with_code_words_is_the_code_sheet():
    pages = {0: page(*block("A1.0", "COVER SHEET"), line("Governing Codes", 751, 889), line("2015 Michigan Residential Code", 751, 904))}
    assert picks([read_row(0, "A1.0", "COVER SHEET", "cover")], pages)["code"] == [0]


def test_cover_with_a_text_layer_and_no_code_words_is_not_read():
    pages = {0: page(*block("C-1", "COVER SHEET"), line("Community Facility", 751, 889))}
    assert picks([read_row(0, "C-1", "COVER SHEET", "cover")], pages)["code"] == []


def test_cover_with_no_text_layer_cannot_be_ruled_out():
    assert picks([read_row(0, "T1", "COVER SHEET", "cover")])["code"] == [0]


def test_life_safety_sheets_also_hold_code_notes_when_no_code_sheet_exists():
    got = picks([read_row(0, "A1", "SITE PLAN", "site_plan"), read_row(1, "A18-1", "LIFE SAFETY COMPOSITE FIRST FLOOR PLAN", "floor_plan")])
    assert got["code"] == [1] and got["life_safety"] == [1]


@pytest.mark.parametrize("title", ["LIFE SAFETY PLAN", "LIFE-SAFETY PLAN", "EXITING PLAN", "EGRESS PLAN", "CODE AND EXIT PLAN"])
def test_life_safety_by_title(title):
    assert picks([read_row(0, "G3", title, "floor_plan")])["life_safety"] == [0]


def test_a_plan_that_only_names_an_exit_is_not_a_life_safety_sheet():
    assert picks([read_row(0, "A2", "EXIT STAIR ENLARGED PLAN", "floor_plan")])["life_safety"] == []


def test_title_on_the_text_layer_counts_when_the_read_is_lost():
    pages = {0: page(*block("G003", "LIFE SAFETY PLAN"))}
    assert picks([read_row(0, read_ok=False)], pages)["life_safety"] == [0]


def test_text_index_needs_no_spark_read():
    pages = {0: page(*block("A1.0", "COVER"), *[line(f"A{n}.0  Sheet {n}", 100, 200 + 20 * n, 18.9) for n in range(1, 5)]), **{n: page(*block(f"A{n}.0", f"Sheet {n}")) for n in range(1, 5)}}
    got = picks([read_row(i) for i in range(5)], pages)
    assert got["index"] == {"text": [0], "spark": [], "cover": []}


def test_a_second_index_that_repeats_the_first_is_not_read():
    first = [line(f"A{n}.0  Sheet {n}", 100, 200 + 20 * n, 18.9) for n in range(1, 5)]
    pages = {0: page(*block("A1.0", "COVER"), *first), 1: page(*block("A9.0", "NOTES"), *first), **{n: page(*block(f"A{n}.0", f"Sheet {n}")) for n in range(2, 5)}}
    assert picks([read_row(i) for i in range(5)], pages)["index"]["text"] == [0]


def test_index_that_continues_on_a_second_sheet_is_read_there_too():
    pages = {0: page(*block("A1.0", "COVER"), *[line(f"A{n}.0  Sheet {n}", 100, 200 + 20 * n, 18.9) for n in range(1, 5)]),
             1: page(*block("A9.0", "NOTES"), *[line(f"B{n}.0  Sheet {n}", 100, 200 + 20 * n, 18.9) for n in range(1, 5)]),
             **{n: page(*block(f"A{n}.0", f"Sheet {n}")) for n in range(2, 5)}, **{n + 4: page(*block(f"B{n}.0", f"Other {n}")) for n in range(1, 5)}}
    assert picks([read_row(i) for i in range(9)], pages)["index"]["text"] == [0, 1]


def test_index_sheet_with_no_text_goes_to_spark_by_title():
    assert picks([read_row(0, "A1", "COVER", "cover"), read_row(1, "T1-3", "INDEX OF DRAWINGS")])["index"] == {"text": [], "spark": [1], "cover": []}


def test_cover_is_tried_for_an_index_when_no_title_names_one():
    assert picks([read_row(0, "A1", "COVER SHEET", "cover")])["index"] == {"text": [], "spark": [], "cover": [0]}


def test_set_with_no_cover_and_no_index_title_reads_no_index():
    assert picks([read_row(0, "", "", "floor_plan")])["index"] == {"text": [], "spark": [], "cover": []}


@pytest.mark.parametrize("title, kind", [("DOOR SCHEDULE", "door"), ("DOOR SCHEDULE (CONT.)", "door"), ("FINISH SCHEDULE", "finish"), ("FINISH PLAN, LEGEND & SCHEDULE", "finish"), ("ROOM SCHEDULE", "finish"), ("WINDOW SCHEDULE", "window")])
def test_schedule_sheet_with_no_text_by_title(title, kind):
    assert picks([read_row(0, "A5", title, "schedule")])["schedules"] == [[0, kind]]


def test_door_and_window_schedule_on_one_sheet_is_both():
    assert picks([read_row(0, "A111", "DOOR & WINDOW SCHEDULE", "schedule")])["schedules"] == [[0, "door"], [0, "window"]]


def test_equipment_schedule_is_not_a_door_schedule():
    assert picks([read_row(0, "A3-1", "EQUIPMENT SCHEDULE", "schedule")])["schedules"] == []


def test_schedule_on_the_text_layer_is_not_sent_to_spark():
    assert picks([read_row(0, "A5", "DOOR SCHEDULE", "schedule")], {0: page(*block("A5", "DOOR SCHEDULE"), *DOORS)})["schedules"] == []


def test_title_that_names_a_schedule_the_text_layer_could_not_parse_goes_to_spark():
    pages = {0: page(*block("A400", "DOOR SCHEDULE & DETAILS"), line("SCHEDULE &", 2780, 1982, 26.9))}
    assert picks([read_row(0, "A400", "DOOR SCHEDULE & DETAILS", "schedule")], pages)["schedules"] == [[0, "door"]]


def test_heading_the_text_layer_could_not_parse_goes_to_spark_whatever_the_title_says():
    pages = {0: page(*block("A2.1", "FLOOR PLAN"), line("GROUND LEVEL DOOR SCHEDULE", 100, 200, 18.9), line("TAG", 100, 240, 9.6))}
    assert picks([read_row(0, "A2.1", "FLOOR PLAN", "floor_plan")], pages)["schedules"] == [[0, "door"]]


def test_a_note_that_mentions_a_schedule_is_not_a_heading():
    pages = {1: page(*block("A2.0", "FOUNDATION PLAN"), line("Refer to window schedule for details.", 100, 200))}
    assert picks([read_row(1, "A2.0", "FOUNDATION PLAN", "floor_plan")], pages)["schedules"] == []


def test_a_legend_line_at_body_size_is_not_a_heading():
    pages = {1: page(*block("A012", "EXITING PLAN"), line("NEW DOOR - SEE DOOR SCHEDULE", 100, 200, 9.6))}
    assert picks([read_row(1, "A012", "EXITING PLAN", "floor_plan")], pages)["schedules"] == []


@pytest.mark.parametrize("cells", [1, 2], ids=["one-line", "number-and-title"])
def test_a_row_of_the_sheet_index_is_not_a_heading(cells):
    sheets = [("G0.1", "COVER SHEET"), ("A1.1", "FIRST FLOOR PLAN"), ("A6.1", "DOOR SCHEDULE"), ("A6.3", "ROOM FINISH SCHEDULE")]
    index = [x for n, (number, title) in enumerate(sheets)
             for x in ([line(f"{number}  {title}", 640, 300 + 24 * n, 12)] if cells == 1 else [line(number, 640, 300 + 24 * n, 12, width=40), line(title, 720, 300 + 24 * n, 12)])]
    pages = {0: page(*block(*sheets[0]), *index), **{n: page(*block(*s)) for n, s in enumerate(sheets[1:], 1)}}
    assert [s for s in picks([read_row(i) for i in range(4)], pages)["schedules"] if s[0] == 0] == []   # the cover is not sent for a strip read


def test_schedule_class_with_only_a_title_block_on_the_text_layer_cannot_be_placed():
    pages = {0: page(*block("A6", "DETAILS & SCHEDULES"))}
    assert picks([read_row(0, "A6", "DETAILS & SCHEDULES", "schedule")], pages)["unplaced"] == [0]


def test_schedule_class_with_a_full_text_layer_was_searched_so_it_is_placed():
    pages = {0: page(*block("E7", "ELECTRICAL SCHEDULES"), line("x" * 500, 100, 200), *[line("y" * 500, 100, 200 + n) for n in range(4)])}
    assert picks([read_row(0, "E7", "ELECTRICAL SCHEDULES", "schedule")], pages)["unplaced"] == []


def test_unread_sheet_with_no_text_cannot_be_placed():
    got = picks([read_row(0, "A1", "COVER", "cover"), read_row(1, read_ok=False)])
    assert got["unplaced"] == [1]


def test_schedule_class_with_a_parsed_schedule_is_placed():
    pages = {0: page(*block("A6.0", "SCHEDULES"), *DOORS)}
    assert picks([read_row(0, "A6.0", "SCHEDULES", "schedule")], pages)["unplaced"] == []


def test_sheet_that_cannot_be_placed_is_named_on_the_stage(tmp_path):
    ctx = pdf_review(tmp_path, lambda p, i: (p.insert_text((2400, 1560), "DETAILS & SCHEDULES", fontsize=12.6), p.insert_text((2420, 1620), "A6", fontsize=31.5)), [read_row(0, "A6", "DETAILS & SCHEDULES", "schedule")])
    meta.set_transport(Model({}))
    out = declared_stage(ctx)
    assert any(row["what"] == "Sheets not placed" and "A6" in row["why"] for row in out["cannot_tell"])


def test_schedule_class_with_a_generic_title_and_no_text_cannot_be_placed():
    assert picks([read_row(0, "A6", "DETAILS & SCHEDULES", "schedule")])["unplaced"] == [0]


def test_unread_sheet_with_a_text_layer_is_placed_by_its_text():
    assert picks([read_row(0, read_ok=False)], {0: page(*block("A1.0", "COVER"))})["unplaced"] == []


def tagged(*marks, kind="circle", size=(24, 24)):
    return [[m, 300 + 40 * i, 300, [*size, kind]] for i, m in enumerate(marks)]


def test_door_tags_come_from_the_plan_on_the_schedules_own_sheet():
    pages = {1: page(*block("A2.1", "MAIN FLOOR PLAN"), *DOORS, tags=tagged("1", "2", "3"))}
    tags = picks([read_row(1, "A2.1", "MAIN FLOOR PLAN", "floor_plan")], pages)["tags"]
    assert (tags["text"], tags["spark"]) == ([[1, 0, [1]]], [])
    assert [t[:3] for t in tags["found"]] == [[1, "door", "1"], [1, "door", "2"], [1, "door", "3"]]


def test_schedule_on_its_own_sheet_serves_every_plan_of_its_series_with_its_tags():
    pages = {0: page(*block("A6.0", "DETAILS & SCHEDULES"), *DOORS), 1: page(*block("A2.0", "FIRST FLOOR PLAN"), tags=tagged("1", "2")),
             2: page(*block("A2.1", "SECOND FLOOR PLAN"), tags=tagged("3")), 3: page(*block("B2.0", "FLOOR PLAN"), tags=tagged("1")), 4: page(*block("A2.2", "DEMOLITION FLOOR PLAN"), tags=tagged("1"))}
    assert picks([read_row(i) for i in range(5)], pages)["tags"]["text"] == [[0, 0, [1, 2]]]


def test_door_schedule_with_no_tags_on_any_text_plan_sends_the_storeys_to_spark():
    pages = {0: page(*block("A6.0", "DETAILS & SCHEDULES"), *DOORS), 1: page(*block("A2.0", "FIRST FLOOR PLAN"))}
    got = picks([read_row(0, "A6.0"), read_row(1, "A2.0", "FIRST FLOOR PLAN", "floor_plan", "FIRST FLOOR")], pages, {"order": [1]})
    assert got["tags"]["spark"] == [1]


def test_spark_tags_fall_back_on_plans_without_a_floors_stage():
    rows = [read_row(0, "A400", "DOOR SCHEDULE", "schedule"), read_row(1, "A101", "RENOVATION FLOOR PLAN", "floor_plan"), read_row(2, "AD101", "DEMOLITION FLOOR PLAN", "floor_plan"),
            read_row(3, "A121", "REFLECTED CEILING PLAN", "floor_plan"), read_row(4, "A721", "FURNITURE PLAN", "floor_plan"), read_row(5, "E101", "LIGHTING PLAN", "floor_plan")]
    assert picks(rows)["tags"]["spark"] == [1]


def test_keynote_circles_on_a_schedules_own_plan_are_not_its_tags():
    doors = table("DOOR SCHEDULE", ["MARK", "SIZE", "SWING"], [["101A", "36", "LEFT"], ["102A", "36", "RIGHT"], ["103B", "60", "BIFOLD"]])
    pages = {1: page(*block("A2.1", "MAIN FLOOR PLAN"), *doors, tags=tagged("1", "2", "3", "4", "5"))}
    tags = picks([read_row(1, "A2.1", "MAIN FLOOR PLAN", "floor_plan", "MAIN")], pages, {"order": [1]})["tags"]
    assert (tags["text"], tags["found"], tags["spark"]) == ([], [], [1])


def test_short_tags_that_share_nothing_with_the_schedule_on_their_plan_go_to_spark():
    doors = table("DOOR SCHEDULE", ["MARK", "SIZE", "SWING"], [["101A", "36", "LEFT"]])
    pages = {1: page(*block("A2.1", "MAIN FLOOR PLAN"), *doors, tags=tagged("KN1", "KN2"))}
    tags = picks([read_row(1, "A2.1", "MAIN FLOOR PLAN", "floor_plan", "MAIN")], pages, {"order": [1]})["tags"]
    assert (tags["text"], tags["found"], tags["spark"]) == ([], [], [1])


def test_schedule_whose_own_plan_shares_no_mark_pairs_with_another_plan():
    pages = {1: page(*block("A2.1", "MAIN FLOOR PLAN"), *DOORS, tags=tagged("7", "8")), 2: page(*block("A2.2", "UPPER FLOOR PLAN"), tags=tagged("1", "2"))}
    tags = picks([read_row(1, "A2.1", "MAIN FLOOR PLAN", "floor_plan"), read_row(2, "A2.2", "UPPER FLOOR PLAN", "floor_plan")], pages)["tags"]
    assert tags["text"] == [[1, 0, [2]]]


def schedule_of(kind, *marks):
    return {"kind": kind, "cols": ["MARK"], "rows": [{"MARK": m} for m in marks]}


def test_window_tags_are_the_largest_polygon_or_box_family_that_holds_a_mark():
    tags = [["7", 0, 0, [30, 30, "poly"]], ["8", 0, 0, [30, 30, "poly"]], ["9", 0, 0, [30, 30, "poly"]], ["A", 0, 0, [20, 14, "poly"]], ["B", 0, 0, [21, 13, "poly"]], ["C", 0, 0, [20, 14, "rect"]], ["A", 0, 0, [12, 12, "circle"]]]
    assert [t[3] for t in declared._family(tags, schedule_of("window", "A", "B", "C"))] == [[20, 14, "poly"], [21, 13, "poly"]]


def test_window_tags_keep_the_family_with_more_members_when_two_hold_marks():
    tags = [["A", 0, 0, [20, 14, "rect"]], ["B", 0, 0, [30, 30, "poly"]], ["C", 0, 0, [30, 30, "poly"]]]
    assert [t[0] for t in declared._family(tags, schedule_of("window", "A", "B"))] == ["B", "C"]


def test_window_schedule_with_no_mark_in_a_polygon_or_box_has_no_family():
    assert declared._family([["A", 0, 0, [12, 12, "circle"]]], schedule_of("window", "A")) == []


def test_door_tags_are_the_digit_family_that_covers_most_marks():
    tags = [["7", 0, 0, [18, 18, "rect"]], ["1", 0, 0, [24, 24, "circle"]], ["2", 0, 0, [24, 24, "circle"]], ["3", 0, 0, [24, 23, "circle"]], ["4", 0, 0, [24, 25, "circle"]]]
    got = declared._family(tags, schedule_of("door", "1", "2", "3"))
    assert [t[0] for t in got] == ["1", "2", "3"]   # one px shorter is the same family, and one px taller is not


def test_door_tags_skip_a_family_with_a_word_in_it():
    tags = [["1", 0, 0, [24, 24, "circle"]], ["KN", 0, 0, [24, 24, "circle"]], ["2", 0, 0, [18, 18, "rect"]]]
    assert [t[0] for t in declared._family(tags, schedule_of("door", "1", "2"))] == ["2"]


def test_door_tags_never_come_from_a_polygon():
    assert declared._family([["1", 0, 0, [24, 24, "poly"]]], schedule_of("door", "1")) == []


def test_finish_schedule_on_a_plan_sheet_reads_rooms_there():
    finish = table("FINISH SCHEDULE", ["NO.", "NAME", "FLOOR"], [["101", "KITCHEN", "TILE"]])
    pages = {2: page(*block("A2.1", "MAIN FLOOR PLAN"), *finish, line("KITCHEN", 300, 300), line("101", 305, 320))}
    assert picks([read_row(2, "A2.1", "MAIN FLOOR PLAN", "floor_plan")], pages)["rooms"] == {"text": [2], "spark": []}


def test_finish_schedule_with_no_text_uses_the_room_numbers_in_a_plans_text():
    # USC draws its finish schedule as lines and keeps its plan's room numbers in the text layer
    pages = {0: page(*block("A2", "FIRST FLOOR PLAN"), line("KITCHEN", 300, 300), line("102", 305, 320))}
    got = picks([read_row(0, "A2", "FIRST FLOOR PLAN", "floor_plan", "FIRST FLOOR"), read_row(1, "A4-2", "FINISH SCHEDULE", "schedule")], pages, {"order": [0]})
    assert got["rooms"] == {"text": [0], "spark": []}


def test_finish_schedule_with_no_text_sends_the_storeys_to_spark_for_rooms():
    got = picks([read_row(0, "A2", "FIRST FLOOR PLAN", "floor_plan", "FIRST FLOOR"), read_row(1, "A4-2", "FINISH SCHEDULE", "schedule")], floors={"order": [0]})
    assert got["rooms"] == {"text": [], "spark": [0]}


def test_tiles_overlap_and_stay_on_the_sheet():
    assert tiles(2, 2) == [pytest.approx(c) for c in [(0, 0, 0.54, 0.54), (0, 0.46, 0.54, 1), (0.46, 0, 1, 0.54), (0.46, 0.46, 1, 1)]]
    inner = tiles(3, 3, box=(0.2, 0.1, 0.8, 0.7))
    assert min(c[0] for c in inner) == pytest.approx(0.2) and max(c[3] for c in inner) == pytest.approx(0.7)
