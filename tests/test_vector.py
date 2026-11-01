import json
import sys
import time
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from storey import pipeline, vector
from storey.meta import Meter
from storey.pipeline import Context

QUARTER = "1/4\" = 1'-0\""
PT_PER_IN = 1.5   # at 1/4 in = 1 ft
K = 0.5523        # control point distance for a quarter circle drawn as one Bezier
WIDTH, HEIGHT = 1224, 792
PX_PER_PT = 2.0   # the fixture's sheet records say the image is twice the page size
PLAN_BOX = [300, 300, 1640, 1140]   # sheet pixels, around the walls
STAIR_BOX = (580 * PX_PER_PT, 300 * PX_PER_PT, 650 * PX_PER_PT, 420 * PX_PER_PT)   # free-standing in page 11's right room, with no shaft walls drawn
EIGHTH = "1/8\" = 1'-0\""
HALF_X, HALF_Y = 100, 50            # where page 13's half-size copy sits
HOSTILE_NOTE = "1/" + "9" * 310 + "\" = 1'-0\""   # the read copies a printed scale exactly, and this one parses to infinity


def draw_door(shape, hinge, shut, opened):
    """A 36 in swing door: the leaf line plus a quarter arc from the shut end to the open end."""
    (hx, hy), (sx, sy), (ox, oy) = hinge, shut, opened
    shape.draw_bezier(shut, (sx + (ox - hx) * K, sy + (oy - hy) * K), (ox + (sx - hx) * K, oy + (sy - hy) * K), opened)
    shape.finish(color=(0, 0, 0), width=0.3)
    shape.draw_line(hinge, opened)
    shape.finish(color=(0, 0, 0), width=0.3)


def draw_plan(page, doors=(420, 500, 560, 610, 655, 700), dx=0, dy=0, shut=(), dims=(), layer=0):
    """A row of rooms inside a double-line wall. Each entry of `doors` is a partition with one door, and the door's height changes
    from partition to partition so that a shifted copy lines up in one way only. `layer` is the dimension lines' optional content group."""
    wall = 1.26
    shape = page.new_shape()

    def line(a, b, width=wall, oc=0):
        shape.draw_line((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy))
        shape.finish(color=(0, 0, 0), width=width, oc=oc)

    for x0, y0, x1, y1 in ((180, 180, 780, 540), (189, 189, 771, 531)):
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            line(a, b)
    for n, x in enumerate(doors):
        top = 240 + 37 * n
        for face in (x - 3, x + 3):
            line((face, 189), (face, top))
            line((face, top + 54), (face, 531))
        line((x - 3, top), (x + 3, top))
        line((x - 3, top + 54), (x + 3, top + 54))
        if n in shut:   # a wall drawn across the opening, so these rooms do not connect
            line((x + 3, top), (x + 3, top + 54))
        else:
            draw_door(shape, (x + 3 + dx, top + dy), (x + 3 + dx, top + 54 + dy), (x + 57 + dx, top + dy))
    for text, inches, y in dims:
        length = inches * PT_PER_IN + 6   # the line overruns its ticks by 3 pt at each end, as on the Capitol set
        line((180 - 3, y), (180 - 3 + length, y), 0.12, layer)
        page.insert_text((180, y - 4), text, fontsize=8)
    line((860, 177), (860, 177 + 240 * PT_PER_IN + 6), 0.12, layer)
    page.insert_text((856 + dx, 400 + dy), "20'-0\"", fontsize=8, rotate=90)
    shape.commit()


GOOD = (("33'-4\"", 400, 600), ("16'-8\"", 200, 620), ("8'-4\"", 100, 640))
BAD = (*GOOD, ("10'-2\"", 120, 660))   # drawn 120 in long and labelled 122 in
MANY = (*((f"{feet}'-0\"", feet * 12, 560 + 20 * k) for k, feet in enumerate(range(4, 24, 2))), ("10'-2\"", 120, 760))   # ten that agree and the one of BAD that does not
STRING = (("10'-6\"", 177, 372, 680), ("4'-0\"", 357, 435, 680), ("14'-0\"", 177, 435, 700))


def write_fixture():
    doc = pymupdf.open()
    dim = doc.add_ocg("A-anno-dims")
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), doors=(420,), dims=GOOD, layer=dim)          # 0 two rooms, one door
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), doors=(420,), dims=BAD, layer=dim)           # 1 one dimension is wrong
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), doors=(420,), dims=GOOD, shut=(0,), layer=dim)   # 2 two rooms, no door
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), dims=GOOD, layer=dim)                        # 3 seven rooms in a row
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), dx=30, dy=-12, layer=dim)                    # 4 the life safety copy of page 3, moved
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), doors=(420,), dims=GOOD, layer=dim)          # 5 rotated below
    doc[5].set_rotation(270)
    draw_hall(doc.new_page(width=WIDTH, height=HEIGHT))                                              # 6 a corridor between two walls
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), doors=(), dims=GOOD, layer=dim)              # 7 outer walls only
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), doors=(420,), dims=BAD)                      # 8 no dimension layer, as in a Revit export
    draw_hall(doc.new_page(width=WIDTH, height=HEIGHT), right=((189, 333), (387, 531)))              # 9 a door in the hall's right wall, level with the label
    add_door(doc[9], (381, 333), (381, 387), (435, 333))
    draw_hall(doc.new_page(width=WIDTH, height=HEIGHT), oc=doc.add_ocg("A-wall-full"))               # 10 walls on a layer, under a heavier pen that draws no wall
    draw_decoy_pen(doc[10])
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), doors=(420,), dims=GOOD, layer=dim)          # 11 a door on the east side of STAIR_BOX, far from the route
    add_door(doc[11], (650, 333), (650, 387), (704, 333))
    draw_notched(doc.new_page(width=WIDTH, height=HEIGHT))                                           # 12 a U-shaped building
    half = doc.new_page(width=WIDTH, height=HEIGHT)                                                  # 13 page 4 at half size, as a life safety sheet at another scale
    half.show_pdf_page(pymupdf.Rect(HALF_X, HALF_Y, HALF_X + WIDTH / 2, HALF_Y + HEIGHT / 2), pymupdf.open("pdf", doc.tobytes()), 4)
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), doors=(420,), dims=GOOD, layer=dim)          # 14 a dimension string 6 in over its overall
    draw_string(doc[14], dim)
    draw_plan(doc.new_page(width=WIDTH, height=HEIGHT), doors=(420,), dims=MANY, layer=dim)          # 15 one dimension wrong among eleven
    draw_hollow(doc.new_page(width=WIDTH, height=HEIGHT))                                            # 16 two rooms inside hollow walls 0.41 m thick
    draw_thick_top(doc.new_page(width=WIDTH, height=HEIGHT))                                         # 17 a hall along x under a wall 0.34 m thick
    draw_hall(doc.new_page(width=WIDTH, height=HEIGHT), lean=0.6)                                    # 18 page 6 with the hall's walls 0.6 pt off square
    draw_wide_hall(doc.new_page(width=WIDTH, height=HEIGHT), bottom=445, stair=True)                 # 19 a hall 97 in wide with a stair 46 in wide standing in it
    draw_wide_hall(doc.new_page(width=WIDTH, height=HEIGHT), bottom=420)                             # 20 the same hall with no stair and its lower wall 25 in higher than printed
    draw_thick_top(doc.new_page(width=WIDTH, height=HEIGHT), bottom=20)                              # 21 page 17 with its lower wall 0.34 m thick too
    draw_thick_opening(doc.new_page(width=WIDTH, height=HEIGHT))                                     # 22 an opening with no door arc in a wall 0.71 m thick, level with the hall label
    draw_wide_hall(doc.new_page(width=WIDTH, height=HEIGHT), bottom=445, stair=True, landing=True)   # 23 page 19 with a landing that stands further into the hall than its flight
    draw_thick_top(doc.new_page(width=WIDTH, height=HEIGHT), bottom=20, lean=-2, label=475)          # 24 page 21 with its walls 2 pt off square, so the upper wall's face steps a pixel under the label
    return doc.tobytes(garbage=4, deflate=True, no_new_id=True)


STAIR_IN_HALL = (300, 300, 500, 369)   # points: a flight against the hall's upper wall, as on 710's upper floors
LANDING = (500, 300, 560, 390)         # the landing at its end, as on 2828's upper floors, where it stands further into the hall


def draw_wide_hall(page, bottom, stair=False, landing=False):
    """A hall along x from y 300 to `bottom`, printed 8'-1" at x 560 from the upper wall's centre, with its label at x 388."""
    shape = page.new_shape()
    for rect in ((180, 180, 780, 540), (189, 189, 771, 531)):
        shape.draw_rect(rect)
        shape.finish(color=(0, 0, 0), width=1.26)
    for y in (291, 300, bottom, bottom + 9):
        shape.draw_line((189, y), (771, y))
        shape.finish(color=(0, 0, 0), width=1.26)
    if stair:
        x0, y0, x1, y1 = STAIR_IN_HALL
        shape.draw_rect(STAIR_IN_HALL)
        for x in range(x0 + 18, x1, 18):
            shape.draw_line((x, y0), (x, y1))   # treads, which a ray across the hall runs along
        if landing:
            shape.draw_rect(LANDING)
        shape.finish(color=(0, 0, 0), width=0.6)
    shape.draw_line((560, 292.5), (560, 444))   # 97 in between ticks at 295.5 and 441, plus the 3 pt overrun at each end
    shape.finish(color=(0, 0, 0), width=0.12)
    shape.commit()
    page.insert_text((556, 380), "8'-1\"", fontsize=8, rotate=90)
    page.insert_text((380, 410 if stair else 365), "Hall", fontsize=8)


def draw_thick_top(page, bottom=7, lean=0, label=440):
    """A hall 67 pt wide along x, under a double-line wall 20 pt thick and over one `bottom` pt thick, both running the building's length.
    `lean` moves the right end of each wall line along y, and `label` is where the hall label starts along x."""
    shape = page.new_shape()
    for rect in ((180, 180, 780, 540), (189, 189, 771, 531)):
        shape.draw_rect(rect)
        shape.finish(color=(0, 0, 0), width=1.26)
    for y in (280, 300, 367, 367 + bottom):
        shape.draw_line((189, y), (771, y + lean))
        shape.finish(color=(0, 0, 0), width=1.26)
    shape.commit()
    page.insert_text((label, 340), "Hall", fontsize=8)


def draw_thick_opening(page):
    """A hall 67 pt wide along y, its right wall two 9 pt walls round a 24 pt chase, 42 pt in all, with an opening 50 pt wide and no door arc, capped by jambs, level with the label."""
    shape = page.new_shape()
    for rect in ((180, 180, 780, 540), (189, 189, 771, 531)):
        shape.draw_rect(rect)
        shape.finish(color=(0, 0, 0), width=1.26)
    lines = [(291, 189, 291, 531), (300, 189, 300, 531), (367, 340, 409, 340), (367, 390, 409, 390)]
    lines += [(x, top, x, bottom) for x in (367, 376, 400, 409) for top, bottom in ((189, 340), (390, 531))]
    for x0, y0, x1, y1 in lines:
        shape.draw_line((x0, y0), (x1, y1))
        shape.finish(color=(0, 0, 0), width=1.26)
    shape.commit()
    page.insert_text((322, 368), "Hall", fontsize=8)


def draw_hollow(page):
    """Two rooms and a door, every wall two lines 24 pt apart with nothing between them, as building 3370 is drawn with its hatch taken out."""
    shape = page.new_shape()
    for rect in ((180, 180, 780, 540), (204, 204, 756, 516), (468, 204, 492, 300), (468, 354, 492, 516)):
        shape.draw_rect(rect)
        shape.finish(color=(0, 0, 0), width=1.26)
    draw_door(shape, (492, 300), (492, 354), (546, 300))
    shape.commit()


def draw_string(page, layer):
    """Capitol A2.1's rear stair: the 10'-6" runs 6 in past the landing and overlaps the 4'-0". Each value matches its own drawn length, and each line overruns its ticks by 3 pt."""
    shape = page.new_shape()
    for text, x0, x1, y in STRING:
        shape.draw_line((x0, y), (x1, y))
        shape.finish(color=(0, 0, 0), width=0.12, oc=layer)
        page.insert_text(((x0 + x1) / 2 - 10, y - 4), text, fontsize=8)
    shape.commit()


def draw_hall(page, right=((189, 531),), oc=0, lean=0):
    """A corridor between two double-line walls, 67 pt between faces and 74 pt between centres. `right` lists the runs of its right wall, so a break in them is an opening.
    `lean` moves the bottom end of each of the corridor's wall lines along x, as the walls of a plan turned to square sit on the generated sets."""
    shape = page.new_shape()
    walls = [(180, 180, 780, 540), (189, 189, 771, 531), (300, 189, 300, 531), (307, 189, 307, 531)]
    for x0, y0, x1, y1 in walls + [(x, top, x, bottom) for top, bottom in right for x in (374, 381)]:
        if x0 == x1:
            shape.draw_line((x0, y0), (x1 + lean, y1))
        else:
            shape.draw_rect((x0, y0, x1, y1))
        shape.finish(color=(0, 0, 0), width=1.26, oc=oc)
    shape.draw_line((300.5, 565), (380.5, 565))   # 74 pt between centres, 49 in, plus the 3 pt overrun at each end
    shape.finish(color=(0, 0, 0), width=0.12)
    page.insert_text((330, 561), "4'-1\"", fontsize=8)
    page.insert_text((330, 360), "Hall", fontsize=8)
    shape.commit()


def add_door(page, hinge, shut, opened):
    shape = page.new_shape()
    draw_door(shape, hinge, shut, opened)
    shape.commit()


def draw_decoy_pen(page):
    """Two heavy lines a wall's thickness apart, off the wall layer, around the hall label. The pen rule alone would take these for the walls."""
    shape = page.new_shape()
    for x in (334, 341):
        shape.draw_line((x, 200), (x, 520))
        shape.finish(color=(0, 0, 0), width=2.4)
    shape.commit()


def draw_notched(page):
    """Rooms round a notch that is open to the south, so the notch is ground. A door at its far end opens onto it, and the left room has a door of its own."""
    shape = page.new_shape()
    runs = (((640, 300), (330, 300), (330, 540), (180, 540), (180, 180), (780, 180), (780, 540), (700, 540), (700, 300), (694, 300)),      # outer face, broken at the door
            ((640, 291), (321, 291), (321, 531), (189, 531), (189, 189), (771, 189), (771, 531), (709, 531), (709, 291), (694, 291)),      # inner face
            ((640, 291), (640, 300)), ((694, 291), (694, 300)),                                                                            # jambs
            ((297, 189), (297, 240), (303, 240), (303, 189)), ((297, 531), (297, 294), (303, 294), (303, 531)))                            # partition, broken at its door
    for run in runs:
        for a, b in zip(run, run[1:]):
            shape.draw_line(a, b)
            shape.finish(color=(0, 0, 0), width=1.26)
    draw_door(shape, (640, 300), (694, 300), (640, 354))
    draw_door(shape, (303, 240), (303, 294), (357, 240))
    shape.commit()


@pytest.fixture(scope="session")
def pdf_bytes():
    return write_fixture()


def sheet(index, kind="vector"):
    return {"index": index, "image": f"sheet_{index:03d}.png", "width": WIDTH * 2, "height": HEIGHT * 2, "kind": kind}


def row(index, scale=QUARTER, cls="floor_plan", level="First floor", box=PLAN_BOX):
    return {"index": index, "number": f"A{index}", "title": "", "cls": cls, "level": level, "scale": scale, "plan_box": box, "read_ok": True}


@pytest.fixture
def review(tmp_path, pdf_bytes):
    def make(sheets, rows, **stages):
        path = tmp_path / "r"
        (path / "upload").mkdir(parents=True, exist_ok=True)
        (path / "upload/0000.pdf").write_bytes(pdf_bytes)
        (path / "stage_read.json").write_text(json.dumps({"sheets": rows}))
        for name, value in stages.items():
            (path / f"stage_{name}.json").write_text(json.dumps(value))
        return Context(path, Meter(ledger=tmp_path / "ledger.jsonl"), sheets, False)
    return make


def stair(sheet_index, state="found", point=(1400, 700), box=None):
    """One row of the verticals stage in Task 5's exact fields. A cell's box is optional, because only a located stair has one."""
    cell = {"state": state, "signals": {"spark": "3/3", "crop": "yes", "ink": 1.0, "aligned": True, "reason": ""}, "point": list(point), "crop": "crop_A.jpg", "mask": None}
    return {"id": "A", "name": "Stair 1", "established": True, "pos": list(point), "floors": {str(sheet_index): cell | ({"box": list(box)} if box else {})}}


NO_EXITS = {"exits": {"sheets": []}}


def exits(sheet_index, *markers):
    """The declared stage with one exits row in Task 6's shape, plus its markers. Each marker is x, y in sheet pixels, width in mm, capacity."""
    return {"exits": {"sheets": [{"sheet": sheet_index, "sum": None, "reads": None, "markers": [{"point": [x, y], "width_mm": w, "capacity": c} for x, y, w, c in markers]}]}}


def crossing_at(path, x):
    """The y where a polyline crosses the vertical line at x, in sheet pixels."""
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        if min(x0, x1) <= x <= max(x0, x1) and x0 != x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise AssertionError("the route does not cross that line")


def test_arc_found_with_width(review):
    out = vector.vector_stage(review([sheet(0)], [row(0)]))["sheets"][0]
    assert len(out["doors"]) == 1
    door = out["doors"][0]
    assert door["width_in"] == pytest.approx(36, abs=0.5)
    assert door["hinge"] == pytest.approx([423 * PX_PER_PT, 240 * PX_PER_PT], abs=2)
    assert door["point"] == pytest.approx([423 * PX_PER_PT, 267 * PX_PER_PT], abs=2)   # the middle of the opening


def quarter_arc(cx, cy, r, sweep_deg=90):
    """Strokes of a sampled arc and a leaf from the centre to its first end, as Nx4 points."""
    a = np.radians(np.linspace(0, sweep_deg, 17))
    p = np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
    return np.vstack([np.c_[p[:-1], p[1:]], [cx, cy, *p[0]]])


def test_arc_drawn_on_top_of_itself_is_found():
    # the Capitol set repeats some door arcs four times, which makes every node of the curve look like a junction
    strokes = quarter_arc(100, 100, 54)
    assert len(vector.door_arcs(np.vstack([strokes] * 4), 2 / 3)) == 1


def test_door_width_is_the_radius_on_a_quarter_arc():
    arc = vector.door_arcs(quarter_arc(100, 100, 54), 2 / 3)[0]
    assert vector.door_width(arc) * 2 / 3 == pytest.approx(36, abs=0.1)


def test_door_width_past_a_quarter_turn_comes_from_the_chord():
    # the Capitol and Baxter 32 in doors: an arc through the shut end and the leaf tip with extra bulge, so its radius reads 29.5 in
    r = 32 * 1.5 * np.sqrt(2) / (2 * np.sin(np.radians(50)))
    arc = vector.door_arcs(quarter_arc(100, 100, r, 100), 2 / 3)[0]
    assert r * 2 / 3 == pytest.approx(29.5, abs=0.1)
    assert vector.door_width(arc) * 2 / 3 == pytest.approx(32, abs=0.1)


def test_scale_chain_passes_good_dimensions(review):
    out = vector.vector_stage(review([sheet(0)], [row(0)]))["sheets"][0]
    assert out["scale_in_per_pt"] == pytest.approx(2 / 3)
    assert out["scale_checked"] == 4    # three along the bottom and one rotated


def test_scale_chain_rejects_bad_scale(review):
    out = vector.vector_stage(review([sheet(1)], [row(1)]))["sheets"][1]
    assert out["scale_in_per_pt"] is None
    assert "10'-2\"" in out["skipped"]
    assert out["doors"] == [] and out["routes"] == [] and out["corridors"] == []


def test_one_dimension_in_ten_that_disagrees_keeps_the_scale_and_is_named(review):
    # Abbot X5.1: a value moved off its line onto a leader pairs with the leader, and 46 of 48 strings agree
    out = vector.vector_stage(review([sheet(15)], [row(15)]))["sheets"][15]
    assert out["scale_in_per_pt"] == pytest.approx(2 / 3) and "disagrees" not in out["skipped"] and out["doors"]
    assert out["scale_checked"] == 12 and out["scale_outliers"] == ["10'-2\": drawn 120.0 in, printed 122.0 in"]


def test_wrong_scale_note_on_a_sheet_of_many_strings_is_rejected(review):
    out = vector.vector_stage(review([sheet(15)], [row(15, scale=EIGHTH)]))["sheets"][15]
    assert out["scale_in_per_pt"] is None and "disagrees" in out["skipped"] and out["scale_outliers"] == []


def test_a_scale_one_percent_off_is_dropped_though_only_its_longest_string_misses():
    # a note read 1% wide keeps every short string within an inch and moves only the longest one past it, which one string in ten could excuse: Capitol A2.0 and A2.2, 3370 A1.1 to A1.4
    dims = [dim(f"{n // 12}'-{n % 12}\"", 0, n * PT_PER_IN + 6, at=20 * k) for k, n in enumerate(range(36, 151, 6))] + [dim("33'-4\"", 0, 400 * PT_PER_IN + 6, at=999)]
    assert vector.check_scale(dims, 2 / 3, True) == (21, None)
    assert "33'-4\"" in vector.check_scale(dims, 2 / 3 * 1.01, True)[1]


def test_past_the_string_cap_no_string_is_excused(monkeypatch):
    dims = [dim(f"{f}'-0\"", 0, f * 12 * PT_PER_IN + 6, at=20 * k) for k, f in enumerate(range(4, 24, 2))] + [dim("10'-2\"", 0, 120 * PT_PER_IN + 6, at=999)]
    assert vector.check_scale(dims, 2 / 3, True) == (11, None)
    monkeypatch.setattr(vector, "STRING_MAX_DIMS", 2)
    assert "10'-2\"" in vector.check_scale(dims, 2 / 3, True)[1]


def test_wrong_scale_note_is_rejected(review):
    out = vector.vector_stage(review([sheet(0)], [row(0, scale="1/8\" = 1'-0\"")]))["sheets"][0]
    assert out["scale_in_per_pt"] is None and out["skipped"]


def test_no_dimension_layer_checks_the_median_string_only(review):
    # with no layer the line beside a string may belong to something else, so one odd string says nothing about the scale
    out = vector.vector_stage(review([sheet(8)], [row(8)]))["sheets"][8]
    assert out["scale_in_per_pt"] == pytest.approx(2 / 3)
    assert out["scale_checked"] == 5


def test_no_dimension_layer_still_rejects_a_wrong_scale(review):
    out = vector.vector_stage(review([sheet(8)], [row(8, scale="1/8\" = 1'-0\"")]))["sheets"][8]
    assert out["scale_in_per_pt"] is None and out["skipped"]


def test_scale_without_text_stands_unchecked(review):
    out = vector.vector_stage(review([sheet(4)], [row(4)]))["sheets"][4]   # only the rotated dimension: too few to fit an overrun
    assert out["scale_in_per_pt"] == pytest.approx(2 / 3)
    assert out["scale_checked"] == 0


def test_route_stays_inside_walkable(review):
    ctx = review([sheet(0), sheet(3)], [row(0), row(3, level="Ground")],
                 floors={"order": [3, 0], "discharge": 3}, verticals={"verticals": [stair(0, box=(1300, 400, 1500, 1000))]}, declared=NO_EXITS)
    routes = vector.vector_stage(ctx)["sheets"][0]["routes"]
    assert len(routes) == 1
    route = routes[0]
    assert route["end_source"] == "matrix" and route["to_exit"] == "Stair A"
    assert route["label"] == "Minimum. Examiner to confirm."
    assert 240 * PX_PER_PT < crossing_at(route["path"], 420 * PX_PER_PT) < 294 * PX_PER_PT   # through the door, not through the wall
    # from the far corner of the left room, round the lower door jamb, to the near edge of the stair box
    feet = (np.hypot(417 - 189, 531 - 294) + (650 - 417)) * 2 / 3 / 12
    assert feet * 0.9 < route["length_ft"] < feet * 1.1


def test_route_ends_on_a_stair_known_only_by_its_point(review):
    ctx = review([sheet(0), sheet(3)], [row(0), row(3, level="Ground")],
                 floors={"order": [3, 0], "discharge": 3}, verticals={"verticals": [stair(0, point=(1400, 700))]}, declared=NO_EXITS)
    route = vector.vector_stage(ctx)["sheets"][0]["routes"][0]
    assert route["end_source"] == "matrix" and route["to_exit"] == "Stair A"
    assert route["path"][-1] == pytest.approx([1400, 700], abs=8)
    assert 240 * PX_PER_PT < crossing_at(route["path"], 420 * PX_PER_PT) < 294 * PX_PER_PT
    feet = (np.hypot(417 - 189, 531 - 294) + np.hypot(700 - 417, 350 - 294)) * 2 / 3 / 12   # far corner to the lower door jamb, then to the point
    assert route["length_ft"] == pytest.approx(feet, rel=0.05)


def test_scan_sheet_is_skipped_with_reason(review):
    out = vector.vector_stage(review([sheet(0, "scan")], [row(0)]))["sheets"][0]
    assert "scan" in out["skipped"].lower()
    assert out["doors"] == [] and out["scale_in_per_pt"] is None


def test_sheet_that_is_not_a_plan_is_skipped(review):
    out = vector.vector_stage(review([sheet(0)], [row(0, cls="schedule")]))["sheets"][0]
    assert out["skipped"] and out["doors"] == []


def test_no_scale_is_skipped_with_reason(review):
    out = vector.vector_stage(review([sheet(0)], [row(0, scale="AS SHOWN")]))["sheets"][0]
    assert out["scale_in_per_pt"] is None and "scale" in out["skipped"].lower()


def test_failed_read_stage_skips_every_sheet(review):
    ctx = review([sheet(0)], [])
    (ctx.dir / "stage_read.json").write_text(json.dumps({"failed": "Sheet read did not finish."}))
    out = vector.vector_stage(ctx)["sheets"][0]
    assert out["skipped"] and out["doors"] == []


def test_broken_pdf_marks_the_sheet_and_the_stage_finishes(review):
    ctx = review([sheet(0)], [row(0)])
    (ctx.dir / "upload/0000.pdf").write_bytes(b"%PDF-1.7 this is not a drawing")
    out = vector.vector_stage(ctx)["sheets"][0]
    assert out["skipped"] and out["doors"] == []


def test_pdf_is_read_in_a_child_process(review, monkeypatch):
    monkeypatch.setattr(pymupdf, "open", lambda *a, **k: pytest.fail("the stage opened the upload itself"))
    assert vector.vector_stage(review([sheet(0)], [row(0)]))["sheets"][0]["doors"]


def test_stuck_child_is_killed_and_the_stage_finishes(review, monkeypatch):
    monkeypatch.setattr(vector, "COMMAND", [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(vector, "CHILD_SECONDS", 0.5)
    started = time.monotonic()
    out = vector.vector_stage(review([sheet(0)], [row(0)]))["sheets"][0]
    assert time.monotonic() - started < 10
    assert out["skipped"] and out["doors"] == []


@pytest.mark.parametrize("failure", [lambda: 1 / 0, lambda: ([{"length_ft": float("inf")}], None)], ids=["raises", "json refuses the result"])
def test_failed_route_keeps_the_sheet_and_the_next_one(review, monkeypatch, failure):
    ctx = review([sheet(3)], [row(3)])
    real, calls = vector.longest_route, []

    def fails_once(plan, ends, stairs):
        calls.append(1)
        return failure() if len(calls) == 1 else real(plan, ends, stairs)
    monkeypatch.setattr(vector, "longest_route", fails_once)
    folder = ctx.dir / "vector"
    folder.mkdir()
    job = {"pdf": str(ctx.dir / "upload/0000.pdf"), "plans": [3, 0, 8], "read": [row(3, level="Ground"), row(0), row(8)], "floors": {"order": [3, 0, 8], "discharge": 3},
           "verticals": [stair(0), stair(8)], "exits": [], "sheets": [sheet(3), sheet(0), sheet(8)]}
    vector.measure_all(job, folder)
    assert len(calls) == 2
    failed = json.loads((folder / "0.json").read_text())
    assert failed["doors"] and failed["routes"] == [] and "did not finish" in failed["skipped"]
    assert json.loads((folder / "8.json").read_text())["routes"][0]["end_source"] == "matrix"
    assert json.loads((folder / "3.json").read_text())["doors"]


def test_value_json_refuses_costs_that_sheet_only(review, monkeypatch):
    ctx = review([sheet(0)], [row(0)])
    real = vector.measure

    def poisoned(plan, r):
        real(plan, r)
        plan.entry["scale_in_per_pt"] = float("inf") if r["index"] == 0 else plan.entry["scale_in_per_pt"]
    monkeypatch.setattr(vector, "measure", poisoned)
    folder = ctx.dir / "vector"
    folder.mkdir()
    job = {"pdf": str(ctx.dir / "upload/0000.pdf"), "plans": [0, 3], "read": [row(0), row(3)], "floors": {}, "verticals": [], "exits": [], "sheets": [sheet(0), sheet(3)]}
    vector.measure_all(job, folder)
    assert "could not be measured" in json.loads((folder / "0.json").read_text())["skipped"]
    assert json.loads((folder / "3.json").read_text())["doors"]


def test_hostile_scale_note_costs_that_sheet_only(review):
    out = vector.vector_stage(review([sheet(0), sheet(3)], [row(0, scale=HOSTILE_NOTE), row(3)]))["sheets"]
    assert out[0]["scale_in_per_pt"] is None and "scale" in out[0]["skipped"].lower()
    assert out[3]["doors"]


def test_sheet_that_cannot_be_measured_says_so_and_the_next_still_runs(review, monkeypatch):
    ctx = review([sheet(0), sheet(3)], [row(0), row(3)])
    real = vector.measure
    monkeypatch.setattr(vector, "measure", lambda plan, r: real(plan, r) if r["index"] == 3 else 1 / 0)
    folder = ctx.dir / "vector"
    folder.mkdir()
    job = {"pdf": str(ctx.dir / "upload/0000.pdf"), "plans": [0, 3], "read": [row(0), row(3)], "floors": {}, "verticals": [], "exits": [], "sheets": [sheet(0), sheet(3)]}
    vector.measure_all(job, folder)
    assert "could not be measured" in json.loads((folder / "0.json").read_text())["skipped"]
    assert json.loads((folder / "3.json").read_text())["doors"]


def test_stage_result_survives_the_pipeline(review):
    ctx = review([sheet(0)], [row(0)])
    (ctx.dir / "sheets.json").write_text(json.dumps([sheet(0)]))
    pipeline.run(ctx.dir, stages=[("vector", "Measuring vector sheets", vector.vector_stage)])
    assert json.loads((ctx.dir / "stage_vector.json").read_text())["sheets"]["0"]["doors"]


def test_rotated_page_lands_on_the_render(review):
    # the page turns 270 degrees, so page point (x, y) renders at (y, WIDTH - x) and a sheet is taller than it is wide
    rotated = {**sheet(5), "width": HEIGHT * 2, "height": WIDTH * 2}
    out = vector.vector_stage(review([rotated], [row(5, box=None)]))["sheets"][5]
    assert out["scale_checked"] == 4
    door = out["doors"][0]
    assert door["hinge"] == pytest.approx([240 * PX_PER_PT, (WIDTH - 423) * PX_PER_PT], abs=2)
    assert door["width_in"] == pytest.approx(36, abs=0.5)


def test_no_route_ends_gives_reason(review):
    ctx = review([sheet(0)], [row(0)], floors={"order": [0], "discharge": 0}, verticals={"verticals": [stair(0)]}, declared=NO_EXITS)
    out = vector.vector_stage(ctx)["sheets"][0]
    assert out["routes"] == [] and "exit" in out["skipped"].lower()   # stairs do not end a route on the discharge level
    assert out["doors"]


def test_unknown_discharge_gives_reason(review):
    ctx = review([sheet(0)], [row(0)], floors={"order": [0], "discharge": None}, verticals={"verticals": [stair(0)]}, declared=NO_EXITS)
    out = vector.vector_stage(ctx)["sheets"][0]
    assert out["routes"] == [] and out["skipped"]


def test_only_a_stair_found_on_this_floor_ends_a_route(review):
    ctx = review([sheet(0), sheet(3)], [row(0), row(3, level="Ground")], floors={"order": [3, 0], "discharge": 3},
                 verticals={"verticals": [stair(0, state="missing")]}, declared=NO_EXITS)
    out = vector.vector_stage(ctx)["sheets"][0]
    assert out["routes"] == [] and out["skipped"]


def test_failed_stages_upstream_give_reason(review):
    ctx = review([sheet(0)], [row(0)], floors={"failed": "x"}, verticals={"failed": "x"}, declared={"failed": "x"})
    out = vector.vector_stage(ctx)["sheets"][0]
    assert out["routes"] == [] and out["skipped"]
    assert out["doors"]


def test_low_reach_withholds_routes(review):
    ctx = review([sheet(2), sheet(3)], [row(2), row(3, level="Ground")], floors={"order": [3, 2], "discharge": 3},
                 verticals={"verticals": [stair(2)]}, declared=NO_EXITS)
    out = vector.vector_stage(ctx)["sheets"][2]
    assert out["routes"] == [] and "reach" in out["skipped"].lower()


@pytest.mark.parametrize("rows", [[], [{"sheet": 3, "sum": 147, "reads": [147, 147]}]], ids=["no sheets", "a sheet with no markers"])
def test_exits_with_no_markers_leave_the_stairs_and_the_real_reason(review, rows):
    ctx = review([sheet(0), sheet(3)], [row(0), row(3, level="Ground")], floors={"order": [3, 0], "discharge": 3},
                 verticals={"verticals": [stair(0)]}, declared={"exits": {"sheets": rows}})
    out = vector.vector_stage(ctx)["sheets"]
    assert out[0]["routes"][0]["end_source"] == "matrix"
    assert "discharge level" in out[3]["skipped"]


def test_life_safety_exit_is_moved_onto_the_plan(review):
    # page 4 is page 3 moved by (30, -12) pt. The marker sits in the right room of the life safety sheet.
    ctx = review([sheet(3), sheet(4)], [row(3), row(4, box=[0, 0, 2448, 1584])], floors={"order": [3], "discharge": 3},
                 verticals={"verticals": []}, declared=exits(4, ((740 + 30) * PX_PER_PT, (400 - 12) * PX_PER_PT, 915.0, 180)))
    route = vector.vector_stage(ctx)["sheets"][3]["routes"][0]
    assert route["end_source"] == "life_safety"
    assert route["to_exit"] == "Exit, 915 mm, 180 people"
    assert route["path"][-1] == pytest.approx([740 * PX_PER_PT, 400 * PX_PER_PT], abs=8)
    for n, x in enumerate((420, 500, 560, 610, 655, 700)):
        assert (240 + 37 * n) * PX_PER_PT < crossing_at(route["path"], x * PX_PER_PT) < (294 + 37 * n) * PX_PER_PT


def test_life_safety_sheet_at_another_scale_is_moved_onto_the_plan(review):
    # page 13 is page 4 at half size with its own scale note, so a length on it counts double on page 3
    x, y = (HALF_X + (740 + 30) / 2) * PX_PER_PT, (HALF_Y + (400 - 12) / 2) * PX_PER_PT
    ctx = review([sheet(3), sheet(13)], [row(3), row(13, scale=EIGHTH, box=[0, 0, 2448, 1584])], floors={"order": [3], "discharge": 3},
                 verticals={"verticals": []}, declared=exits(13, (x, y, 915.0, 180)))
    route = vector.vector_stage(ctx)["sheets"][3]["routes"][0]
    assert route["end_source"] == "life_safety"
    assert route["path"][-1] == pytest.approx([740 * PX_PER_PT, 400 * PX_PER_PT], abs=8)


def test_life_safety_markers_replace_stairs(review):
    ctx = review([sheet(0), sheet(3), sheet(4)], [row(0, level="Ground"), row(3), row(4, box=[0, 0, 2448, 1584])], floors={"order": [0, 3], "discharge": 0},
                 verticals={"verticals": [stair(3, box=(420, 400, 500, 1000))]}, declared=exits(4, (770 * PX_PER_PT, 388 * PX_PER_PT, None, 147)))
    assert vector.vector_stage(ctx)["sheets"][3]["routes"][0]["end_source"] == "life_safety"


def test_marker_above_discharge_ends_at_the_stair_it_names(review):
    # the marker is printed beside the stair, so the route stops on reaching the stair and does not walk through it to the marker
    box = (700 * PX_PER_PT, 350 * PX_PER_PT, 760 * PX_PER_PT, 450 * PX_PER_PT)
    ctx = review([sheet(0), sheet(3), sheet(4)], [row(0, level="Ground"), row(3), row(4, box=[0, 0, 2448, 1584])], floors={"order": [0, 3], "discharge": 0},
                 verticals={"verticals": [stair(3, box=box)]}, declared=exits(4, ((730 + 30) * PX_PER_PT, (470 - 12) * PX_PER_PT, None, 147)))
    route = vector.vector_stage(ctx)["sheets"][3]["routes"][0]
    assert route["end_source"] == "life_safety"
    assert route["path"][-1][0] == pytest.approx(box[0], abs=12)


def test_marker_on_the_discharge_level_keeps_its_own_place(review):
    box = (700 * PX_PER_PT, 350 * PX_PER_PT, 760 * PX_PER_PT, 450 * PX_PER_PT)
    ctx = review([sheet(3), sheet(4)], [row(3), row(4, box=[0, 0, 2448, 1584])], floors={"order": [3], "discharge": 3},
                 verticals={"verticals": [stair(3, box=box)]}, declared=exits(4, ((730 + 30) * PX_PER_PT, (470 - 12) * PX_PER_PT, 915.0, 180)))
    route = vector.vector_stage(ctx)["sheets"][3]["routes"][0]
    assert route["path"][-1] == pytest.approx([730 * PX_PER_PT, 470 * PX_PER_PT], abs=8)


def test_hollow_walls_with_no_hatch_still_trace(review):
    # the inside of each double-line wall is free space, which read as rooms nobody reaches: reach fell to 50% on every floor of 3370 with its hatch taken out
    ctx = review([sheet(16)], [row(16)], floors={"order": [16], "discharge": 16}, verticals={"verticals": []}, declared=exits(16, (740 * PX_PER_PT, 400 * PX_PER_PT, 915.0, 180)))
    route = vector.vector_stage(ctx)["sheets"][16]["routes"][0]
    assert 300 * PX_PER_PT < crossing_at(route["path"], 480 * PX_PER_PT) < 354 * PX_PER_PT   # through the door
    feet = (np.hypot(468 - 204, 516 - 354) + np.hypot(740 - 492, 400 - 354)) * 2 / 3 / 12   # far corner of the left room, round the lower jamb, to the exit
    assert route["length_ft"] == pytest.approx(feet, rel=0.1)


def test_one_traced_room_is_not_a_building(review):
    # page 7 has outer walls only, which is what a wrong pen can leave: reach would be 100% and the route short
    ctx = review([sheet(7)], [row(7)], floors={"order": [7], "discharge": 7}, verticals={"verticals": []}, declared=exits(7, (740 * PX_PER_PT, 400 * PX_PER_PT, 915.0, 180)))
    out = vector.vector_stage(ctx)["sheets"][7]
    assert out["routes"] == [] and "rooms" in out["skipped"].lower()


def test_exit_on_a_sheet_that_does_not_line_up_gives_reason(review):
    # page 0 has one door against page 3's six
    ctx = review([sheet(0), sheet(3)], [row(0), row(3)], floors={"order": [3], "discharge": 3}, verticals={"verticals": []}, declared=exits(0, (770 * PX_PER_PT, 388 * PX_PER_PT, None, 180)))
    out = vector.vector_stage(ctx)["sheets"][3]
    assert out["routes"] == [] and "line up" in out["skipped"]


def test_exit_marker_far_from_any_floor_is_dropped(review):
    ctx = review([sheet(3)], [row(3, box=[0, 0, 2448, 1584])], floors={"order": [3], "discharge": 3}, verticals={"verticals": []}, declared=exits(3, (2400, 100, None, 180)))
    out = vector.vector_stage(ctx)["sheets"][3]
    assert out["routes"] == [] and out["skipped"]


def test_plan_box_short_of_the_exit_side_keeps_the_route(review):
    # a read box can stop short of the walls on one side while its pad still holds them, and the exit must stay on the floor
    def route(box):
        ctx = review([sheet(3)], [row(3, box=box)], floors={"order": [3], "discharge": 3}, verticals={"verticals": []}, declared=exits(3, (760 * PX_PER_PT, 400 * PX_PER_PT, 915.0, 180)))
        return vector.vector_stage(ctx)["sheets"][3]["routes"][0]

    full, short = route(PLAN_BOX), route([300, 300, 750 * PX_PER_PT, 1140])   # the walls run to 780 pt
    assert short["length_ft"] == pytest.approx(full["length_ft"], abs=0.2)
    assert short["path"][-1] == pytest.approx([760 * PX_PER_PT, 400 * PX_PER_PT], abs=8)


def test_route_does_not_start_outside_the_plan_box(review):
    # the left room runs from 189 pt, and section marks past the box are drawn in the wall pen, so nothing past the box may start a route
    left = 205
    ctx = review([sheet(3)], [row(3, box=[left * PX_PER_PT, 300, 1640, 1140])], floors={"order": [3], "discharge": 3},
                 verticals={"verticals": []}, declared=exits(3, (740 * PX_PER_PT, 400 * PX_PER_PT, 915.0, 180)))
    assert vector.vector_stage(ctx)["sheets"][3]["routes"][0]["path"][0][0] >= left * PX_PER_PT


def test_route_enters_a_stair_shaft_through_its_door(review):
    # the shaft's walls are not drawn, so only the ring around the box stops the route walking in on the near side
    ctx = review([sheet(3), sheet(11)], [row(3, level="Ground"), row(11)], floors={"order": [3, 11], "discharge": 3},
                 verticals={"verticals": [stair(11, box=STAIR_BOX)]}, declared=NO_EXITS)
    path = vector.vector_stage(ctx)["sheets"][11]["routes"][0]["path"]
    assert max(x for x, _ in path) >= 650 * PX_PER_PT   # round the box to its east side
    assert path[-1][0] == pytest.approx(650 * PX_PER_PT, abs=30) and 333 * PX_PER_PT <= path[-1][1] <= 387 * PX_PER_PT


def test_ground_outside_a_door_does_not_start_a_route(review):
    # the notch on page 12 is inside the footprint and a door opens onto it, and its corner is the farthest walkable point from this exit, but it is not a room
    ctx = review([sheet(12)], [row(12)], floors={"order": [12], "discharge": 12}, verticals={"verticals": []}, declared=exits(12, (230 * PX_PER_PT, 230 * PX_PER_PT, 915.0, 180)))
    x, y = vector.vector_stage(ctx)["sheets"][12]["routes"][0]["path"][0]
    assert x > 709 * PX_PER_PT and y > 291 * PX_PER_PT   # the far corner of the right wing, the next farthest


def test_marker_snaps_to_a_room_not_to_a_speck():
    walk = np.zeros((200, 200), bool)
    walk[100:180, 20:180] = True    # a room
    walk[60, 100] = True            # a sliver of free space inside a wall line, nearer to the marker than the room is
    end = {"name": "Exit", "source": "life_safety", "point": (100, 50), "box": None}
    assert vector.starts_for(walk, [end], lambda p: p) == {(100, 100): end}


def test_ground_outside_a_door_is_not_a_room():
    closed = np.zeros((200, 300), bool)
    closed[20:24, 20:124] = closed[120:124, 20:124] = closed[20:124, 20:24] = closed[20:124, 120:124] = True
    labels = np.zeros((200, 300), np.int32)
    labels[24:120, 24:120] = 1      # walls on four sides
    labels[24:120, 124:220] = 2     # the far side of one wall, open on the other three
    assert np.unique(vector.enclosed_rooms(labels, closed)).tolist() == [0, 1]


def test_marker_with_no_room_on_the_sheet_lands_nowhere():
    walk = np.zeros((200, 200), bool)
    walk[60, 100] = True
    assert vector.starts_for(walk, [{"name": "Exit", "source": "life_safety", "point": (100, 50), "box": None}], lambda p: p) == {}


def test_no_rooms_stays_no_rooms():
    labels = np.zeros((50, 50), np.int32)
    assert not vector.enclosed_rooms(labels, np.zeros((50, 50), bool)).any()


def test_room_name_from_the_drawing_is_cut_short():
    long_name = "Room " + "x" * 500
    plan = vector.Plan(np.zeros((0, 8)), [], [(long_name, 8, 8, 12, 12, True)], 1.0, (0, 0, 20, 20), None, vector.blank())
    assert len(vector.room_name(plan, np.ones((20, 20), np.int32), lambda p: (int(p[0]), int(p[1])), 1)) <= vector.MAX_NAME


def test_stair_shaft_is_closed_except_at_its_door():
    # a 4 m square room with a stair box in it. The shaft's walls are not drawn, so without the ring a route walks straight in.
    walk = np.ones((160, 160), bool)
    box = (60, 60, 100, 100)
    door = ((100, 64), (100, 96))   # hinge and shut end in the east side of the box
    ring = vector.stair_ring(walk.shape, box, [door])
    assert ring.any() and not ring[80, 100 + 2]        # open at the door
    assert ring[80, 60 - 2] and ring[60 - 2, 80]       # closed on the west and north
    assert not vector.stair_ring(walk.shape, box, []).any()   # a stair with no door found stays open, or nothing could reach it


def test_door_clear_of_the_stair_box_leaves_the_stair_open():
    # an open stair in a hall, as on the Capitol set: the box follows the treads, so there is no shaft, and a door a metre away must not seal the stair
    box = (60, 60, 100, 100)
    door = ((112, 110), (140, 110))   # its circle clips the ring's corner, but the door is not in the ring
    assert not vector.stair_ring((200, 200), box, [door]).any()


def test_pen_rule_takes_the_heaviest_pen_that_draws_walls():
    wall = np.array([[0, 0, 400, 0], [0, 8, 400, 8], [0, 80, 400, 80], [0, 88, 400, 88]], float)
    border = np.array([[0, -500, 3000, -500], [0, 900, 3000, 900]], float)   # the heaviest ink on the sheet, but no face has a partner
    thin = np.array([[0, y, 300, y] for y in range(200, 216, 8)], float)
    segs = np.vstack([np.c_[wall, np.full(4, 1.26)], np.c_[border, np.full(2, 2.4)], np.c_[thin, np.full(2, 0.12)]])
    assert vector.wall_pen(segs, 40) == 1.26


def test_wall_layer_names():
    assert vector.is_wall_layer("A-wall-full") and vector.is_wall_layer("X_Leconte 3rd Floor Plan|WALL2")
    assert not vector.is_wall_layer("A-wall-door") and not vector.is_wall_layer("A-WALL-PATT") and not vector.is_wall_layer("A-anno-dims")


def test_wall_layers_beat_the_pen_rule(review):
    # page 10 has a heavier paired pen off the layer, which would read as a wall 7 pt wide across the label
    hall = vector.vector_stage(review([sheet(10)], [row(10)]))["sheets"][10]["corridors"][0]
    assert hall["clear_in"] == pytest.approx(67 * 2 / 3, abs=0.5)


def test_corridor_clear_width(review):
    out = vector.vector_stage(review([sheet(6)], [row(6)]))["sheets"][6]
    assert out["scale_in_per_pt"] == pytest.approx(2 / 3)
    assert len(out["corridors"]) == 1
    hall = out["corridors"][0]
    assert hall["clear_in"] == pytest.approx(67 * 2 / 3, abs=0.5)
    assert hall["printed_in"] == 49   # printed from wall centre to wall centre, as on the Capitol set
    assert hall["point"] == pytest.approx([340 * PX_PER_PT, 357 * PX_PER_PT], abs=30)


@pytest.mark.parametrize("page", [17, 24], ids=["square", "off square"])
def test_a_thick_wall_gives_the_hall_no_false_door(review, page):
    # an even kernel shifts OpenCV's opening and closing by a pixel, so where the face of a wall 0.3 m or thicker steps a pixel, as a wall drawn off square does on page 24
    # and on 3370's fifth floor, the door bridging ran 2 px past it into the hall and the label's ray met a door
    hall = vector.vector_stage(review([sheet(page)], [row(page)]))["sheets"][page]["corridors"][0]
    assert hall["clear_in"] == pytest.approx(67 * 2 / 3, abs=0.5)


def test_an_opening_with_no_arc_in_a_thick_wall_is_still_a_door(review):
    # a door's bridge is as long across as its wall is thick, so dropping every bridge over 0.6 m across dropped this one, the ray ran through it and the hall's 228 in length read as its width
    hall = vector.vector_stage(review([sheet(22)], [row(22)]))["sheets"][22]["corridors"][0]
    assert hall["clear_in"] is None


def test_a_hall_between_two_thick_walls_is_not_a_door(review):
    # each wall's thickness is a run the door bridging keeps, so it bridged the 1.13 m between them and the hall read as one door: CORRIDOR 501 on 3370's fifth floor
    hall = vector.vector_stage(review([sheet(21)], [row(21)]))["sheets"][21]["corridors"][0]
    assert hall["clear_in"] == pytest.approx(67 * 2 / 3, abs=0.5)


def test_walls_a_little_off_square_still_bound_the_hall(review):
    # a wall line 0.6 pt off vertical over its length was left out, so the ray ran on to the next wall: 75% of the wall length on 6267's first floor
    hall = vector.vector_stage(review([sheet(18)], [row(18)]))["sheets"][18]["corridors"][0]
    assert hall["clear_in"] == pytest.approx(67 * 2 / 3, abs=0.5)


def test_a_stair_standing_in_the_hall_leaves_the_width_beside_it(review):
    # 710's upper floors read 97 in from wall to wall where a 46 in flight takes part of the hall, the wrong-pass direction
    box = tuple(v * PX_PER_PT for v in STAIR_IN_HALL)
    ctx = review([sheet(19)], [row(19)], verticals={"verticals": [stair(19, point=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box=box)]})
    hall = vector.vector_stage(ctx)["sheets"][19]["corridors"][0]
    assert hall["clear_in"] == pytest.approx((445 - 369) * 2 / 3, abs=0.5)
    assert hall["printed_in"] is None   # a width printed wall to wall says nothing of the way past a stair


def test_a_landing_wider_than_its_flight_sets_the_width_past_the_stair(review):
    # 2828's upper floors: one ray through the middle of the stair met the flight, and the landing at its end leaves less of the hall
    box = tuple(v * PX_PER_PT for v in (*STAIR_IN_HALL[:2], *LANDING[2:]))
    ctx = review([sheet(23)], [row(23)], verticals={"verticals": [stair(23, point=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box=box)]})
    hall = vector.vector_stage(ctx)["sheets"][23]["corridors"][0]
    assert hall["clear_in"] == pytest.approx((445 - LANDING[3]) * 2 / 3, abs=0.5)


def test_a_printed_width_over_80_in_is_compared(review):
    # a hall printed 8'-1" whose lower wall stands 25 in higher could never read as narrower than printed. The Check stair here has no lines drawn, so it narrows nothing
    ctx = review([sheet(20)], [row(20)], verticals={"verticals": [stair(20, state="check", point=(800, 720))]})
    hall = vector.vector_stage(ctx)["sheets"][20]["corridors"][0]
    assert hall["clear_in"] == pytest.approx(80, abs=0.5) and hall["printed_in"] == 97


def test_door_across_a_hall_label_gives_no_reading(review):
    # the ray through the label runs on through the door to a far wall, and that span is longer than the hall, so the hall's length must not stand in for its width
    hall = vector.vector_stage(review([sheet(9)], [row(9)]))["sheets"][9]["corridors"][0]
    assert hall["clear_in"] is None and hall["printed_in"] is None
    assert hall["point"] == pytest.approx([340 * PX_PER_PT, 357 * PX_PER_PT], abs=30)


def hall_plan(*arcs):
    """Geometry for corridors() on its own, in points, with a label in a hall from x 100 to 167 and y 0 to 600."""
    return vector.Plan(np.zeros((0, 8)), [], [("Hall", 123, 195, 143, 205, True)], 1.0, (0, -10, 300, 610), None, vector.blank(), ipp=2 / 3, arcs=list(arcs))


HALL_WALLS = np.array([[100, 0, 100, 600], [167, 0, 167, 600], [100, 0, 167, 0], [100, 600, 167, 600]], float)
NO_READING = {"point": [133, 200], "clear_in": None, "printed_in": None}


def test_door_along_the_hall_leaves_its_width_alone():
    # the Capitol and Baxter halls: the long ray crosses a door, and the short one is still a good reading
    door = {"c": np.array([113.0, 450.0]), "shut": np.array([154.0, 450.0])}
    walls = np.vstack([HALL_WALLS, [[100, 450, 113, 450], [154, 450, 167, 450]]])
    assert vector.corridors(hall_plan(door), walls, []) == [{"point": [133, 200], "clear_in": 44.7, "printed_in": None}]


def test_a_stair_past_the_wall_narrows_the_hall_by_what_it_puts_in_it():
    # 2828: the landing of a stair open to the hall stands 13 pt past the face of the wall that the label's ray stops at
    plan = hall_plan()
    plan.segs = np.array([[110, 300, 110, 400, 0.6, 0, vector.LINE, 1]], float)
    plan.stairs = [{"point": np.array([75.0, 350.0]), "box": np.array([40.0, 300.0, 110.0, 400.0])}]
    assert vector.corridors(plan, HALL_WALLS, [])[0]["clear_in"] == round(57 * 2 / 3, 1)
    plan.stairs = [{"point": np.array([133.0, 550.0]), "box": None}]   # 710's first floor: the stair's ray has a wall on one side only
    assert vector.corridors(plan, np.vstack([HALL_WALLS[:1], [[167, 0, 167, 500]], HALL_WALLS[2:]]), [])[0]["clear_in"] is None


def test_a_door_beside_a_stair_leaves_the_rays_past_it_to_measure():
    # 3128's lower floors: a door in the hall's wall at the stair, where the ray runs on into the room beyond
    plan = hall_plan()
    plan.segs = np.array([[110, 300, 110, 400, 0.6, 0, vector.LINE, 1]], float)
    plan.stairs = [{"point": np.array([75.0, 350.0]), "box": np.array([40.0, 300.0, 110.0, 400.0])}]
    walls = np.vstack([HALL_WALLS[[0, 2, 3]], [[167, 0, 167, 340], [167, 380, 167, 600], [250, 0, 250, 600]]])
    assert vector.corridors(plan, walls, [])[0]["clear_in"] == round(57 * 2 / 3, 1)
    plan.stairs = [{"point": np.array([75.0, 360.0]), "box": np.array([40.0, 345.0, 110.0, 375.0])}]   # every ray across this stair meets the door
    assert vector.corridors(plan, walls, [])[0]["clear_in"] is None


def test_a_slot_too_narrow_for_a_hall_stays_bridged():
    # two walls 0.5 m thick and 0.4 m apart: the bridge reaches across further than the gap it closes, as a hall's does, but no hall is that narrow
    w = np.zeros((200, 200), bool)
    w[20:180, 40:60] = w[20:180, 76:96] = True
    assert vector.close_doors(w)[100, 60:76].all()


def test_a_tiny_segment_crosses_a_ray_where_it_is_drawn():
    # a segment under 0.5 pt long across the ray counts at any angle, and its line carried on to the ray met it 746 pt from where it is drawn on 710
    assert vector.crossings(np.array([[10.0, 0.0, 10.3, 0.001]]), 0.4, True).tolist() == [10.3]


def test_a_wall_that_crosses_the_ray_outside_the_plan_box_does_not_bound_the_hall():
    # a long wall line under 5 degrees off square touches the box but meets the label's ray outside it, and a hostile page can put that anywhere
    walls = np.vstack([HALL_WALLS[[0, 2, 3]], [[280, -600, 380, 600]]])
    assert vector.corridors(hall_plan(), walls, []) == [NO_READING]


def test_hall_with_walls_on_one_axis_only_gives_no_reading():
    # the walls may be the corridor's two ends, 160 ft apart on the Brandeis sheet, so one bounded axis cannot say which way the width runs
    assert vector.corridors(hall_plan(), HALL_WALLS[:2], []) == [NO_READING]
    assert vector.corridors(hall_plan(), np.zeros((0, 4)), []) == [NO_READING]


def test_hall_on_a_plan_too_large_to_raster_gives_no_reading(monkeypatch):
    monkeypatch.setattr(vector, "MAX_RASTER", 0)
    assert vector.corridors(hall_plan(), HALL_WALLS, []) == [NO_READING]


@pytest.mark.parametrize("note, inches", [
    ("1/4\" = 1'-0\"", 2 / 3), ("1/8\"=1'-0\"", 4 / 3), ("3/16\" = 1'-0\"", 8 / 9), ("SCALE: 1/4\" = 1'-0\"", 2 / 3), ("1/4” = 1’-0”", 2 / 3),
    ("1 1/2\" = 1'-0\"", 12 / 1.5 / 72), ("1\" = 20'-0\"", 240 / 72), ("1:150", 150 / 72), ("1 : 100", 100 / 72), ("SCALE 1:50", 50 / 72)])
def test_scale_notes(note, inches):
    assert vector.inches_per_pt(note) == pytest.approx(inches)


@pytest.mark.parametrize("note", ["", "AS SHOWN", "NTS", "1/0\" = 1'-0\"", "0:50", "1:0", "1/4\" = 0'-0\"", HOSTILE_NOTE, f"{'9' * 400}:1"])
def test_scale_notes_that_give_no_scale(note):
    assert vector.inches_per_pt(note) is None


@pytest.mark.parametrize("text, inches", [("14'-6 1/2\"", 174.5), ("8'-0\"", 96), ("8’-0”", 96), ("A5.0", None), ("3'-9 1/0\"", None)])
def test_feet_and_inches(text, inches):
    assert vector.parse_ftin(text) == inches


def dim(text, start, end, at=100.0, along_x=True):
    """One row of dimensions()."""
    return {"text": text, "inches": vector.parse_ftin(text), "along_x": along_x, "start": float(start), "end": float(end), "at": float(at)}


def strings(*dims, px_per_pt=1.0):
    return vector.dimension_strings(list(dims), vector.Plan(np.zeros((0, 8)), [], [], px_per_pt, (0, 0, 9, 9), None, vector.blank()))


OVERALL = dim("14'-0\"", 0, 258)                # 168 in at 1.5 pt per inch, plus 3 pt of overrun at each end
MEET = dim("7'-0\"", 0, 132, at=60), dim("7'-0\"", 126, 258, at=60)   # neighbours that meet overlap by the two overruns
MEETING = tuple(dim("1'-0\"", x, x + 24, at=500) for x in (1000, 1018, 1036, 1054))   # neighbours far from the rest, so the sheet meets at -6 pt


def test_string_over_its_overall_is_recorded():
    assert strings(OVERALL, dim("10'-6\"", 0, 195, at=60), dim("4'-0\"", 180, 258, at=60)) == [
        {"parts": ["10'-6\"", "4'-0\""], "sum_in": 174, "overall": "14'-0\"", "box": [0, 60 - vector.BESIDE_PT, 258, 60 + vector.BESIDE_PT]}]


def test_string_that_adds_up_is_not_recorded():
    assert strings(OVERALL, *MEET) == []


def test_string_with_one_value_wrong_is_recorded():
    assert strings(OVERALL, dim("7'-0\"", 0, 132, at=60), dim("7'-3\"", 126, 258, at=60))[0]["sum_in"] == 171


@pytest.mark.parametrize("second, recorded", [("7'-0 1/2\"", False), ("7'-1\"", True), ("6'-11 1/2\"", False), ("6'-11\"", True)], ids=["half over", "one over", "half under", "one under"])
def test_a_miss_of_half_an_inch_is_rounding(second, recorded):
    assert bool(strings(OVERALL, dim("7'-0\"", 0, 132, at=60), dim(second, 126, 258, at=60))) is recorded


@pytest.mark.parametrize("start, end, recorded", [(0, 258, True), (0.7, 258, True), (0, 257.3, True), (0, 258.7, True), (0.8, 258, False), (0, 257.2, False), (0, 258.8, False), (1.5, 258, False), (3, 255, False)],
                         ids=["both meet", "start under half an inch short", "end under half an inch short", "end under half an inch past", "start over half an inch short",
                              "end over half an inch short", "end over half an inch past", "start an inch short", "both stop short"])
def test_a_string_must_meet_both_ends_of_its_overall(start, end, recorded):
    """Both ends, to under 1/2 in by the overall's own drawn length, here 0.77 pt. A string that stops shorter than that and prints that much less is drawn correctly."""
    assert bool(strings(OVERALL, dim("10'-6\"", start, 195, at=60), dim("4'-0\"", 180, end, at=60))) is recorded


def test_ends_and_holes_are_measured_in_inches_at_any_scale():
    """At 1/8" = 1'-0" an inch is 0.75 pt, under the 1.5 pt that held at 1/4". The Capitol and Baxter pairs that stop 1 in short of an 8'-1" and print 1 in less are drawn correctly."""
    whole = dim("8'-1\"", 0, 78.75)   # 97 in at 0.75 pt per inch, plus 3 pt of overrun at each end
    assert strings(whole, dim("4'-0\"", 0, 42, at=60), dim("4'-0\"", 36, 78, at=60), *MEETING) == []
    assert strings(whole, dim("4'-0\"", 0, 42, at=60), dim("4'-0\"", 36.75, 78.75, at=60), *MEETING) == []
    assert strings(whole, dim("4'-0\"", 0, 42, at=60), dim("4'-3\"", 36, 78.75, at=60), *MEETING)[0]["sum_in"] == 99


def test_a_small_value_on_a_long_line_keeps_the_point_limit():
    """A 0'-1" paired with a 258 pt line would allow 129 pt at each end, and a 0'-0" gives no scale at all. The part here starts 3 pt in."""
    parts = dim("10'-6\"", 3, 195, at=60), dim("4'-0\"", 180, 258, at=60)
    assert strings(dim("0'-1\"", 0, 258), *parts) == [] and strings(dim("0'-0\"", 0, 258), *parts) == []


def test_a_parts_line_that_runs_past_both_ends_of_its_overall_still_holds_its_string():
    """Capitol A2.1: the line under the 14'-0" carries on past both of its ends, as the 8'-7" there does past the 4'-0"."""
    line = dim("2'-0\"", -42, 6, at=60), dim("10'-6\"", 0, 195, at=60), dim("4'-0\"", 180, 258, at=60), dim("8'-7\"", 252, 412, at=60)
    assert [s["parts"] for s in strings(OVERALL, *line)] == [["10'-6\"", "4'-0\""]]


def test_a_line_along_the_other_axis_holds_no_string():
    assert strings(OVERALL, dim("10'-6\"", 0, 195, at=60, along_x=False), dim("4'-0\"", 180, 258, at=60, along_x=False)) == []


def test_a_sheet_past_the_cap_has_no_strings(monkeypatch):
    monkeypatch.setattr(vector, "STRING_MAX_DIMS", 2)
    assert strings(OVERALL, dim("10'-6\"", 0, 195, at=60), dim("4'-0\"", 180, 258, at=60)) == []


def test_one_part_is_no_string():
    assert strings(OVERALL, dim("13'-0\"", 0, 258, at=60)) == []


def test_parts_on_two_lines_are_not_one_string():
    assert strings(OVERALL, dim("10'-6\"", 0, 195, at=60), dim("4'-0\"", 180, 258, at=70)) == []


def test_parts_a_point_apart_are_one_line():
    assert len(strings(OVERALL, dim("10'-6\"", 0, 195, at=60), dim("4'-0\"", 180, 258, at=61))) == 1


def test_a_gap_past_the_overlap_limit_is_no_string():
    """The sheet meets at -6 pt, and this part runs 25 pt over the next one."""
    assert strings(OVERALL, dim("11'-0\"", 0, 205, at=60), dim("4'-0\"", 180, 258, at=60), *MEETING) == []


def test_a_hole_in_a_string_is_no_string():
    """Baxter X1.0: two ticks 4.2 pt apart inside a party wall, so the parts do not span the overall and their sum need not reach it. Neighbours elsewhere set what meeting looks like."""
    hole = dim("6'-9\"", 0, 127.8, at=60), dim("7'-0\"", 126, 258, at=60)
    assert strings(OVERALL, *hole, *(dim("1'-0\"", x, x + 24, at=500) for x in (0, 18, 36, 54))) == []


def test_what_meeting_looks_like_comes_from_the_sheet():
    """BCOE and the ADU sets end a dimension line at its tick, so their neighbours meet at 0 pt and a gap of 3 pt is a hole there too."""
    flat = [dim("1'-0\"", x, x + 18, at=500) for x in (0, 18, 36, 54)]
    wrong = dim("6'-9\"", 0, 126, at=60), dim("7'-0\"", 126, 252, at=60)
    whole = dim("14'-0\"", 0, 252)
    assert len(strings(whole, *wrong, *flat)) == 1
    assert strings(whole, dim("6'-9\"", 0, 124, at=60), dim("7'-0\"", 127, 252, at=60), *flat) == []


def test_two_neighbours_on_a_sheet_keep_an_observed_gap():
    """A midpoint of -15 and -6 would read the -6 as a hole."""
    assert len(strings(OVERALL, dim("10'-6\"", 0, 195, at=60), dim("4'-0\"", 180, 258, at=60), dim("7'-3\"", 0, 132, at=40), dim("7'-0\"", 126, 258, at=40))) == 2


def test_a_string_printed_twice_is_one_row():
    assert len(strings(OVERALL, *(dim(text, a, b, at=at) for at in (60, 500) for text, a, b in (("10'-6\"", 0, 195), ("4'-0\"", 180, 258))))) == 1


def test_bold_text_printed_on_top_of_itself_is_one_part():
    part = dim("10'-6\"", 0, 195, at=60)
    assert strings(OVERALL, OVERALL, part, part, dim("4'-0\"", 180, 258, at=60))[0]["sum_in"] == 174


def test_a_string_along_y_has_its_box_across_the_line():
    rows = [dim(text, a, b, at, along_x=False) for text, a, b, at in (("14'-0\"", 10, 268, 700), ("10'-6\"", 10, 205, 660), ("4'-0\"", 190, 268, 660))]
    assert strings(*rows, px_per_pt=2.0)[0]["box"] == [2 * (660 - vector.BESIDE_PT), 20, 2 * (660 + vector.BESIDE_PT), 536]


def test_a_sheet_with_no_neighbours_has_no_strings():
    assert strings(OVERALL) == [] and strings() == []


def recorded(ctx):
    return vector.vector_stage(ctx)["sheets"][14]


def test_bad_string_reaches_the_stage_result(review):
    entry = recorded(review([sheet(14)], [row(14)], **NO_EXITS))
    assert entry["scale_in_per_pt"] and [(s["parts"], s["sum_in"], s["overall"]) for s in entry["strings"]] == [(["10'-6\"", "4'-0\""], 174, "14'-0\"")]
    x0, y0, x1, y1 = entry["strings"][0]["box"]
    assert (x0, x1) == (177 * PX_PER_PT, 435 * PX_PER_PT) and y0 < 680 * PX_PER_PT < y1


@pytest.mark.parametrize("scale, reason", [("", "No drawing scale"), (EIGHTH, "disagrees")], ids=["no note", "wrong note"])
def test_a_string_needs_no_scale(review, scale, reason):
    entry = recorded(review([sheet(14)], [row(14, scale=scale)], **NO_EXITS))
    assert reason in entry["skipped"] and entry["scale_in_per_pt"] is None and len(entry["strings"]) == 1


def test_a_sheet_the_stage_does_not_open_has_no_strings(review):
    assert recorded(review([sheet(14)], [row(14, cls="elevation")], **NO_EXITS))["strings"] == []
