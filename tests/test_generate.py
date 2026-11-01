"""The test set generator, on a small building made here, so the tests need neither the dataset nor the network. What Spark reads comes from the model."""
import json

import numpy as np
import pymupdf
import pytest
from shapely import affinity
from shapely.geometry import MultiLineString, box

from eval import defects, draw, dry, generate, measure, model, swiss
from storey import declared, meta

FLOORS = 5
ANGLE = 17
NO_SITE_PLAN = "No site plan in the set"   # shortcut: check 1 now asks for a site plan and these sets draw none, so a clean set raises this one true item, stop leaving it out if the generator draws one


def _row(floor, z, kind, subtype, g, unit="", height=2.0, sill=0.0):
    g = affinity.rotate(g, ANGLE, origin=(0, 0))
    return {"floor_id": str(floor), "plan_id": "1", "unit_id": unit, "unit_usage": "RESIDENTIAL", "entity_type": kind, "entity_subtype": subtype, "geometry": g.wkt, "elevation": str(z + sill), "height": str(height)}


def _floor(k, hall, enclosed=False):
    """A block 10 m deep: a hall and stair in the middle, a unit either side, and a street door on the first floor. `hall` is the hall's width in metres. `enclosed` walls the stair
    off from the hall past a 1 m opening, because a stair standing in a narrow hall leaves too little way past it."""
    z = k * 2.9
    left, right = 6.2, 6.2 + hall      # the hall's wall faces
    end = right + 6
    rows = []
    walls = [box(0, 0, end + 0.3, 0.3), box(0, 9.7, end + 0.3, 10), box(0, 0, 0.3, 10), box(end, 0, end + 0.3, 10), box(left - 0.2, 0, left, 10), box(right, 0, right + 0.2, 10), box(0, 5, left - 0.2, 5.15), box(right + 0.2, 5, end, 5.15)]
    if enclosed:
        walls += [box(left, 5, left + 0.35, 5.15), box(right - 0.35, 5, right, 5.15)]
    for g in walls:
        rows.append(_row(k, z, "separator", "WALL", g, height=2.6))
    for subtype, g, unit in (("BEDROOM", box(0.3, 0.3, 6, 5), "a"), ("LIVING_ROOM", box(0.3, 5.15, 6, 9.7), "a"), ("BEDROOM", box(right + 0.2, 0.3, end, 5), "b"), ("LIVING_ROOM", box(right + 0.2, 5.15, end, 9.7), "b")):
        rows.append(_row(k, z, "area", subtype, g, unit))
    rows.append(_row(k, z, "area", "STAIRCASE", box(left, 0.3, right, 9.7)))
    rows.append(_row(k, z, "feature", "STAIRS", box(right - 1.3, 5.3, right - 0.1, 9.5)))
    rows.append(_row(k, z, "feature", "TOILET", box(1, 6, 1.6, 6.8), "a"))
    for g, unit in ((box(6, 7.55, 6.2, 8.45), "a"), (box(right, 7.55, right + 0.2, 8.45), "b")):
        rows.append(_row(k, z, "opening", "ENTRANCE_DOOR", g, unit))
    for g, unit in ((box(2.5, 5, 3.4, 5.15), "a"), (box(right + 2.5, 5, right + 3.4, 5.15), "b")):
        rows.append(_row(k, z, "opening", "DOOR", g, unit))
    if k == 0:
        rows.append(_row(k, z, "opening", "ENTRANCE_DOOR", box((left + right) / 2 - 0.5, 0, (left + right) / 2 + 0.5, 0.3)))
    for g, unit in ((box(2, 0, 3.5, 0.3), "a"), (box(2, 9.7, 3.5, 10), "a"), (box(right + 2, 0, right + 3.5, 0.3), "b"), (box(right + 2, 9.7, right + 3.5, 10), "b")):
        rows.append(_row(k, z, "opening", "WINDOW", g, unit, height=1.5, sill=0.9))
    return rows


ROWS = [row for k in range(FLOORS) for row in _floor(k, 3.8)]
NARROW_HALL = [row for k in range(FLOORS) for row in _floor(k, 1.7, enclosed=True)]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(swiss, "rows", lambda building: ROWS)
    monkeypatch.setattr(generate, "OUT", None)

    def refuse(kind, **request):
        raise AssertionError("A test reached the network")

    meta.set_transport(refuse)
    yield
    meta.set_transport(None)


def _set(tmp_path, kinds=(), name="set.pdf"):
    m = model.load("7", 1)
    rows = [defects.plant(m, kind) for kind in kinds]
    pdf = tmp_path / name
    return m, rows, draw.draw_set(m, pdf), pdf


def test_clean_model_agrees_with_itself():
    m = model.load("7", 1)
    assert len(m.floors) == FLOORS and [f.name for f in m.floors][:2] == ["FIRST FLOOR", "SECOND FLOOR"]
    for f in m.floors:
        assert {d.mark for d in f.doors if d.tag} == {r["mark"] for r in m.door_rows[f.number]}
        assert {r.number for r in [*f.rooms, f.hall]} == {r["no"] for r in m.finish_rows[f.number]}
    assert m.load_total == {"area": sum(r["area"] for r in m.load_rows), "load": sum(r["load"] for r in m.load_rows)}
    assert m.code["load"] == str(m.load_total["load"]) and m.code["stories"] == str(FLOORS)
    assert sum(r["qty"] for r in m.window_rows) == sum(len(f.windows) for f in m.floors)
    assert [d.kind for d in m.floors[0].doors].count("exit") == 1
    assert m.exit_rows[0]["load"] == m.load_total["load"]


def test_walls_run_along_the_sheet_after_turning():
    m = model.load("7", 1)
    edges = [(a, b) for f in m.floors for p in f.walls for a, b in zip(p.exterior.coords, p.exterior.coords[1:])]
    assert all(abs(a[0] - b[0]) < 0.02 or abs(a[1] - b[1]) < 0.02 for a, b in edges)


def test_a_seed_repeats_and_another_differs(tmp_path):
    a, b, c = model.load("7", 1), model.load("7", 1), model.load("7", 2)
    assert a.name == b.name and [d.hand for f in a.floors for d in f.doors] == [d.hand for f in b.floors for d in f.doors]
    assert [d.hand for f in a.floors for d in f.doors] != [d.hand for f in c.floors for d in f.doors] or a.name != c.name


def test_a_set_repeats_byte_for_byte(tmp_path):
    assert _set(tmp_path, name="a.pdf")[3].read_bytes() == _set(tmp_path, name="b.pdf")[3].read_bytes()


def test_every_sheet_reads_as_a_set(tmp_path):
    m, _, _, pdf = _set(tmp_path)
    with pymupdf.open(pdf) as doc:
        assert len(doc) == 1 + 2 * FLOORS + 3
        assert all((p.rect.width, p.rect.height) == (2592, 1728) for p in doc)
        assert all(draw.GENERATED in p.get_text() and m.date in p.get_text() and m.name.upper() in p.get_text() for p in doc)
        assert draw.CREDIT in doc[0].get_text() and "000 Nonesuch Way" in doc[0].get_text()
        pages = {i: {"w": p.rect.width, "h": p.rect.height, "lines": declared.page_lines(p), "tags": declared.shape_tags(p)} for i, p in enumerate(doc)}
    reads = declared.text_reads(pages)
    assert [r["number"] for r in reads.values()] == [s.number for s in m.sheets]
    assert [r["title"] for r in reads.values()] == [s.title for s in m.sheets]
    assert [r["number"] for r in reads[0]["index"]] == [s.number for s in m.sheets]
    assert [(t["kind"], len(t["rows"])) for t in reads[0]["tables"]] == [("occupant", FLOORS + 1), ("exit", len(m.exit_rows)), ("exit", len(m.corridor_rows))]
    assert reads[0]["headings"] == [] and not any(r["tables"] for r in list(reads.values())[1:])
    door_sheet = next(i for i, s in enumerate(m.sheets) if s.kind == "door")
    titles = [x for x in pages[door_sheet]["lines"] if x["t"].endswith("FLOOR DOOR SCHEDULE")]
    assert len(titles) == FLOORS and len({round(x["y0"]) for x in titles[:3]}) == 1   # one table a floor, standing level as an office lays them out


def test_dimension_strings_match_what_is_drawn(tmp_path):
    from storey import vector

    m, _, _, pdf = _set(tmp_path)
    plan_sheet = next(i for i, s in enumerate(m.sheets) if s.kind == "plan")
    with pymupdf.open(pdf) as doc:
        page = doc[plan_sheet]
        segs, layers = vector.strokes(page)
        dims = vector.dimensions(vector.text_lines(page), segs, layers, (0, 0, 2592, 1728))
    assert "A-WALL" in layers and any("DIMS" in name for name in layers)
    ipp = vector.inches_per_pt(m.scale[1])
    checked, problem = vector.check_scale(dims, ipp, True)
    assert checked >= 8 and problem is None


def test_each_kind_changes_the_model(tmp_path):
    for kind in set(defects.KINDS) - {"corridor", "route"}:
        m, rows, anchors, _ = _set(tmp_path, (kind,), f"{kind}.pdf")
        assert rows[0]["check"] == defects.KINDS[kind][1]
        assert rows[0]["anchor"] in anchors[rows[0]["sheet"]]
        clean = model.load("7", 1)
        assert (json.dumps(m.code) != json.dumps(clean.code) or m.load_total != clean.load_total or m.door_rows != clean.door_rows or m.markers != clean.markers
                or [(s.number, s.on_index, s.block_title) for s in m.sheets] != [(s.number, s.on_index, s.block_title) for s in clean.sheets]
                or [(f.stair_on_plan, len(f.plan_walls), [r.name for r in f.rooms], [bool(d.tag) for d in f.doors]) for f in m.floors] != [(f.stair_on_plan, 0, [r.name for r in f.rooms], [bool(d.tag) for d in f.doors]) for f in clean.floors]), kind


def test_what_the_geometry_cannot_hold_is_refused():
    for kind in ("corridor", "route"):   # this hall is wider than check 14 compares, and no one-stair block is long enough for check 13
        with pytest.raises(defects.NotPlantable):
            defects.plant(model.load("7", 1), kind)


def test_corridor_narrower_than_its_printed_width(tmp_path, monkeypatch):
    monkeypatch.setattr(swiss, "rows", lambda building: NARROW_HALL)
    m, rows, _, pdf = _set(tmp_path, ("corridor",))
    findings = dry.review(m, pdf, tmp_path / "review")["houston"]["findings"]
    assert [f["title"] for f in findings if f["check"] == 14 and f["severity"] == "check"] == ["Corridor on FIFTH FLOOR is narrower than its printed width"]


def test_clean_set_raises_nothing_but_what_the_rules_say(tmp_path):
    m, _, _, pdf = _set(tmp_path)
    by_city = dry.review(m, pdf, tmp_path / "review")
    raised = {f["check"] for f in by_city["houston"]["findings"] if f["severity"] != "unknown" and f["title"] != NO_SITE_PLAN}
    assert raised == {11}, [f["title"] for f in by_city["houston"]["findings"]]
    titles = {city: [f["title"] for f in res["findings"] if f["check"] == 11] for city, res in by_city.items()}
    assert "Too few stairs on FOURTH FLOOR" in titles["houston"] and "Too few stairs on FIFTH FLOOR" in titles["houston"]
    assert not any(t.startswith("Too few") for t in titles["seattle"]) and "One stair here depends on conditions" in titles["seattle"]


def test_titles_wrap_at_one_size(tmp_path):
    m, _, _, pdf = _set(tmp_path)
    with pymupdf.open(pdf) as doc:
        for sheet, page in zip(m.sheets, doc):
            assert " ".join(x["t"] for x in declared.page_lines(page) if x["x0"] > draw.STRIP and x["size"] == 16) == sheet.title


def test_planted_defects_are_raised_from_the_text_layer_and_vectors(tmp_path):
    kinds = "stair_mid", "stair_discharge", "index_extra", "sheet_unlisted", "floor_name", "stories", "edition", "area_total", "door_type", "tag_no_row", "row_no_tag", "room_name"
    m, rows, _, pdf = _set(tmp_path, kinds)
    findings = dry.review(m, pdf, tmp_path / "review")["houston"]["findings"]
    numbers = {i: s.number for i, s in enumerate(s for s in m.sheets if s.in_set)}
    for row in rows:
        hits = [f for f in findings if f["check"] == row["check"] and numbers.get(f["sheet"]) == row["sheet"] and row["phrase"] in f["title"]]
        assert hits, row["kind"]
        assert hits[0]["severity"] in row["severity"], row["kind"]


def test_truth_hash_shows_a_later_change(tmp_path, monkeypatch):
    monkeypatch.setattr(generate, "OUT", tmp_path)
    folder = generate.make("7", 1, "stories")
    path = folder / "truth.json"
    assert generate.verify(path)
    truth = json.loads(path.read_text())
    assert truth["defects"][0]["kind"] == "stories" and truth["defects"][0]["check"] == 5 and len(truth["defects"][0]["box"]) == 4
    assert truth["px_per_pt"] == {"stage files": 6000 / 2592, "review.json": 2000 / 2592}
    path.write_text(json.dumps(truth | {"seed": 2}))
    assert not generate.verify(path)


def test_clean_truth_has_no_defects_and_names_the_rule_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(generate, "OUT", tmp_path)
    truth = json.loads((generate.make("7", 1, "clean") / "truth.json").read_text())
    assert truth["defects"] == []
    outcomes = {row["jurisdiction"]: [f["outcome"] for f in row["floors"]] for row in truth["check_11"]}
    assert outcomes["seattle"] == ["one stair depends on conditions"] * 4
    assert outcomes["houston"] == ["one stair depends on conditions"] * 2 + ["too few stairs"] * 2


def test_second_cycle_changes_only_the_sheets_it_says(tmp_path, monkeypatch):
    monkeypatch.setattr(generate, "OUT", tmp_path)
    monkeypatch.setattr(generate, "MIXED", ("stair_mid", "stories", "tag_no_row", "door_type"))
    monkeypatch.setattr(generate, "CORRECTED", ("stair_mid", "stories"))
    first = generate.make("7", 1, "mixed")
    second = generate.make("7", 1, "second")
    truth = json.loads((second / "truth.json").read_text())
    states = {c["kind"]: c["state"] for c in truth["comments"]}
    assert states == {"stair_mid": "Looks resolved", "stories": "Looks resolved", "tag_no_row": "Still open", "door_type": "Still open", "partition": "New change"}
    assert {c["sheet"] for c in truth["clouds"]} <= set(truth["changed_sheets"])
    with pymupdf.open(first / "set.pdf") as a, pymupdf.open(second / "set.pdf") as b:
        assert len(a) == len(b)
        numbers = [s["number"] for s in truth["sheets"]]
        for number, p, q in zip(numbers, a, b):
            same = p.get_pixmap(dpi=20).samples == q.get_pixmap(dpi=20).samples
            assert same == (number not in truth["changed_sheets"]), number


def test_a_stair_box_as_wide_as_two_flights_is_a_u_stair():
    flights, landings = model._flights([box(0, 0, 4.2, 2.4)])
    assert len(flights) == 2 and len(landings) == 1
    assert model._rect(flights[0].poly)[3] == pytest.approx(1.14)
    flights, landings = model._flights([box(0, 0, 3.8, 1.2)])
    assert len(flights) == 1 and not landings


def test_public_rooms_are_no_unit(tmp_path, monkeypatch):
    top = str(FLOORS - 1)
    public = [r | {"unit_id": "", "unit_usage": "PUBLIC"} if r["floor_id"] == top and r["unit_id"] == "b" else r for r in ROWS]
    public.append(_row(top, (FLOORS - 1) * 2.9, "area", "CORRIDOR", box(10.3, 9.2, 15.5, 9.6)) | {"unit_usage": "PUBLIC"})
    monkeypatch.setattr(swiss, "rows", lambda building: public)
    m, _, _, pdf = _set(tmp_path)
    rooms = m.floors[-1].rooms
    assert {r.unit for r in rooms} == {"", f"{FLOORS}01"} and "CORRIDOR" in {r.name for r in rooms} and "ENTRY" not in {r.name for r in rooms}
    with pymupdf.open(pdf) as doc:
        assert f"UNITS:\n{2 * FLOORS - 1}\n" in doc[0].get_text()
        assert "UNIT \n" not in doc[FLOORS].get_text()


def test_a_pair_of_doors_prints_one_width_everywhere():
    door = model.Door("1001", np.zeros(2), 1.85, np.zeros(2), np.array([1.85, 0]), np.array([0, 1.85]), "LEFT", "exit", 2.1)
    assert model.door_row(door)["size"].startswith('(2) 36" x') and model._leaf_in(door) * 2 == 72


def _ink_boxes(page):
    """Text lines as boxes the height of their capitals, because a line's own box runs into the line above it."""
    rows = []
    for x in declared.page_lines(page):
        rows.append((x["t"], x["x0"] + 1, x["y0"] + 1, x["x1"] - 1, x["y1"] - 1) if x["dir"][1] else (x["t"], x["x0"] + 1, (x["y0"] + x["y1"]) / 2 - 0.36 * x["size"], x["x1"] - 1, (x["y0"] + x["y1"]) / 2 + 0.36 * x["size"]))
    return rows


def test_no_text_overprints_text(tmp_path, monkeypatch):
    monkeypatch.setattr(swiss, "rows", lambda building: NARROW_HALL)
    m, _, _, pdf = _set(tmp_path)
    assert m.floors[1].hall_width
    with pymupdf.open(pdf) as doc:
        for sheet, page in zip(m.sheets, doc):
            if sheet.kind in ("plan", "life"):
                rows = _ink_boxes(page)
                assert not [(a[0], b[0]) for k, a in enumerate(rows) for b in rows[k + 1:] if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]], sheet.number


def test_a_hall_string_slides_its_text_off_a_stair():
    faces = MultiLineString([[(0, 0), (10, 0)], [(0, 3), (10, 3)]])
    (a, b), at = model._string(faces, box(4.6, 1.4, 5.0, 3), np.array([5.2, 1.5]), np.array([0.0, 1.0]), ((0, 0),), (-8, 0), 3.0, 59)
    assert a[0] == b[0] == pytest.approx(5.2) and at[1] == pytest.approx(0.9)


def test_an_exit_marker_stays_off_the_stair():
    taken = [[100, 100, 300, 300]]
    c = type("Canvas", (), {"anchors": {}, "box": lambda *a, **k: None, "lines": lambda *a, **k: None, "circle": lambda *a, **k: None, "text": lambda *a, **k: None})()
    draw.exit_marker(c, {"name": "STAIR 1", "point": (200, 200), "width": 44, "capacity": 146}, lambda p: p, 0, taken)
    x0, y0, x1, y1 = c.anchors["marker:0"]
    assert x0 >= 300 or x1 <= 100 or y0 >= 300 or y1 <= 100


def test_a_cloud_on_text_keeps_to_its_row_and_the_title_strip():
    title = generate._cloud({"sheet": "A1.2", "kind": "floor_name", "box": [draw.STRIP + 9, 1394, draw.W - draw.MARGIN - 9, 1478]})
    bulge = 0.4 * title["scallop"]
    assert title["box"][0] - bulge > draw.STRIP and title["box"][2] + bulge < draw.W - draw.MARGIN
    row = generate._cloud({"sheet": "G0.1", "kind": "stories", "box": [70, 380, 300, 400]})
    assert row["box"][1] - 0.4 * row["scallop"] > 370
    assert generate._cloud({"sheet": "A1.3", "kind": "stair_mid", "box": [900, 700, 1100, 800]})["box"][0] == 886


def test_cloud_scallops_stand_out_evenly():
    doc = pymupdf.open()
    c = draw.Canvas(doc, {name: doc.add_ocg(name) for name in draw.LAYERS})
    x0, y0, x1, y1 = 100, 100, 400, 126
    draw.revision_cloud(c, {"box": [x0, y0, x1, y1], "scallop": 14}, 1)
    c.done()
    mids = [(p.x + 3 * a.x + 3 * b.x + q.x) / 8 for p, a, b, q in [i[1:] for d in doc[0].get_drawings() for i in d["items"] if i[0] == "c"]], \
        [(p.y + 3 * a.y + 3 * b.y + q.y) / 8 for p, a, b, q in [i[1:] for d in doc[0].get_drawings() for i in d["items"] if i[0] == "c"]]
    rise = [max(x0 - x, x - x1, y0 - y, y - y1) for x, y in zip(*mids)]
    assert len(rise) > 40 and min(rise) > 0.3 * 14 and max(rise) < 0.4 * 14


def test_a_title_anchor_holds_its_lines_only(tmp_path):
    m, _, anchors, _ = _set(tmp_path)
    height = {s.kind: anchors[s.number]["title"][3] - anchors[s.number]["title"][1] for s in m.sheets}
    assert height["plan"] < 25 and 35 < height["life"] < 45   # FIRST FLOOR PLAN fits one line, and FIRST FLOOR LIFE SAFETY PLAN wraps onto a second


def test_measure_refuses_an_older_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(measure, "DATA", tmp_path / "reviews")
    folder = tmp_path / "7_1_clean"
    folder.mkdir()
    (folder / "set.pdf").write_bytes(b"regenerated")
    (tmp_path / "reviews/7_1_clean/upload").mkdir(parents=True)
    (tmp_path / "reviews/7_1_clean/upload/0000.pdf").write_bytes(b"first")
    with pytest.raises(SystemExit, match="another upload"):
        measure._run(folder, None)
