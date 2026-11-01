import base64
import hashlib
import json
import threading

import cv2
import httpx
import numpy as np
import openai
import pytest

from storey import meta, verticals
from storey.meta import BudgetReached, Meter
from storey.pipeline import Context
from storey.verticals import ANS, EXTRA, FLOOR_P, GROUND, QUESTION, SCHEMA, WHOLE_P, cluster, decide, floor_index, ink_ratio, verticals_stage

W, H = 3000, 2000            # a sheet's pixels, so the stage must convert: its 2000 px copy is 2/3 of this
FRAME = (960, 640)           # stack pixels: 0.32 of a sheet pixel
PLACE = (900, 600)           # sheet pixels, on the 0-1000 grid 300, 300
SECOND = (2100, 1400)        # far enough from PLACE to be its own row
SHAFTS = "Elevators and shafts are not followed."


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("a test reached the real client"))
    yield
    meta.set_transport(None)


# Drawings


def blank():
    return np.full((H, W, 3), 255, np.uint8)


def stair_at(im, x, y):
    """Parallel treads, 120 sheet pixels square."""
    for dy in range(-60, 61, 12):
        cv2.line(im, (x - 60, y + dy), (x + 60, y + dy), (0, 0, 0), 3)
    return im


def room_at(im, x, y):
    """Walls that keep a place from reading blank but leave the 60 px ink box empty."""
    cv2.rectangle(im, (x - 65, y - 65), (x + 65, y + 65), (0, 0, 0), 3)
    return im


def stair(*places):
    im = blank()
    for x, y in places or (PLACE,):
        stair_at(im, x, y)
    return im


def room(x=PLACE[0], y=PLACE[1]):
    return room_at(blank(), x, y)


def thin_stair():
    """Treads further apart than stair(), so the ink ratio sits between 0.40 and 0.90, as on many real cells."""
    im = blank()
    for dy in range(-60, 61, 18):
        cv2.line(im, (PLACE[0] - 60, PLACE[1] + dy), (PLACE[0] + 60, PLACE[1] + dy), (0, 0, 0), 3)
    return im


def disc_on(im):
    """Marks a sheet so a stub can tell its crop from the others."""
    return cv2.circle(im, (PLACE[0] + 150, PLACE[1] + 150), 15, (0, 0, 0), -1)


# The review folder


def review(tmp_path, images, *, order=None, transforms=None, unaligned=(), reference=None, frame=FRAME, cap=8.0, levels=None):
    """A review with the read and floors stages done. Images are bottom to top, and sheet numbers are not positions, so a mix-up of the two shows."""
    order = order or [10 + 3 * k for k in reversed(range(len(images)))]
    sheets, rows = [], []
    for k, (i, im) in enumerate(zip(order, images)):
        im = im.copy()
        im[k * 7 + 3::53, k * 11 + 5::53] = 200   # pale dots of its own, too faint to count as ink, so no two floors share a crop and a cached reply
        cv2.imwrite(str(tmp_path / f"sheet_{i}.png"), im)
        sheets.append({"index": i, "image": f"sheet_{i}.png", "width": im.shape[1], "height": im.shape[0], "kind": "scan"})
        rows.append({"index": i, "number": f"A{i}", "title": "", "cls": "floor_plan", "level": levels[k] if levels else f"LEVEL {k}", "scale": "", "plan_box": [0, 0, im.shape[1], im.shape[0]], "read_ok": True})
    reference = order[len(order) // 2] if reference is None else reference
    moves = {reference: np.eye(3)} | {i: np.eye(3) for i in order if i not in unaligned} | {order[k]: np.array(T, float) for k, T in (transforms or {}).items()}
    (tmp_path / "stack.mp4").write_bytes(b"stack")
    (tmp_path / "stage_read.json").write_text(json.dumps({"sheets": rows}))
    (tmp_path / "stage_floors.json").write_text(json.dumps({
        "order": order, "reference": reference, "transforms": {str(i): T.tolist() for i, T in moves.items()}, "inliers": {str(i): 99 for i in order},
        "unaligned": [i for i in unaligned], "stack": "stack.mp4", "frame_size": list(frame), "discharge": None, "not_checked": None}))
    return Context(tmp_path, Meter(cap_usd=cap, ledger=tmp_path / "ledger.jsonl"), sheets, False)


def stage(ctx):
    out = verticals_stage(ctx)
    json.dumps(out, allow_nan=False)   # what pipeline.write_json does with it
    return out


# The model


def chat(reply, finish="stop"):
    return {"text": reply if isinstance(reply, str) else json.dumps(reply), "input": 100, "cached": 0, "output": 50, "finish": finish}


def grid(x, y, w=W, h=H):
    return round(x * 1000 / w), round(y * 1000 / h)


def listing(*stairs, floors=3, shafts=()):
    """One compare reply. A stair is a map from floor position to (x, y) in sheet pixels. A floor not in the map is listed as absent."""
    def row(name, at):
        return {"name": name, "floors": [{"floor": f"F{k + 1}", "present": k in at, "x": grid(*at[k])[0] if k in at else 0, "y": grid(*at[k])[1] if k in at else 0} for k in range(floors)]}
    return {"stairs": [row(f"Stair {n}", at) for n, at in enumerate(stairs)],
            "shafts": [{"kind": "elevator", **row(f"Lift {n}", at)} for n, at in enumerate(shafts)]}


def on(floors, place=PLACE):
    return {k: place for k in floors}


def tile(request):
    """The first image of a chat request, decoded."""
    url = request["messages"][-1]["content"][1]["image_url"]["url"]
    return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)


def looks(request):
    """A blind reader: yes when treads fill the middle of the crop."""
    im = tile(request)
    h, w = im.shape
    return {"answer": "yes" if (im[h // 2 - 60:h // 2 + 60, w // 2 - 60:w // 2 + 60] < 128).mean() > 0.1 else "no", "seen": "x"}


def has_disc(request):
    return tile(request)[700:740, 670:710].mean() < 128   # the disc is 100 copy pixels from the stair, 300 on the crop, and its centre snaps 10 up


def no_reply_on_the_disc(request):
    return "" if has_disc(request) else looks(request)


WHOLE = {"objects": [{"object_name": "stair", "x_min": 400, "y_min": 400, "x_max": 600, "y_max": 600}]}


class Model:
    """Stands in for the API. Compare replies are dealt in call order, because the three requests are identical.
    `crop` and `whole` are a reply, or a function of the request. `sam` is the output text, or an exception to raise."""

    def __init__(self, *runs, crop=looks, whole=WHOLE, sam=""):
        self.runs, self.crop, self.whole, self.sam = list(runs), crop, whole, sam
        self.calls, self._lock = [], threading.Lock()

    def __call__(self, kind, **request):
        with self._lock:
            self.calls.append((kind, request))
            if kind == "upload":
                return "file-1"
            if kind == "sam":
                if isinstance(self.sam, Exception):
                    raise self.sam
                return self.sam
            text = request["messages"][-1]["content"][0]["text"]
            if text.startswith("These images"):
                return self.reply(self.runs.pop(0) if len(self.runs) > 1 else self.runs[0], request)
            return self.reply(self.crop if text.startswith("This is a crop") else self.whole, request)

    @staticmethod
    def reply(reply, request):
        if isinstance(reply, Exception):
            raise reply
        return chat(reply(request) if callable(reply) else reply) if not isinstance(reply, dict) or "text" not in reply else reply

    def of(self, start):
        return [r for k, r in self.calls if k == "chat" and r["messages"][-1]["content"][0]["text"].startswith(start)]

    @property
    def sam_prompts(self):
        return [r["input"][0]["content"][0]["text"] for k, r in self.calls if k == "sam"]


def model(*runs, **kw):
    m = Model(*runs, **kw)
    meta.set_transport(m)
    return m


def refused():
    return openai.BadRequestError("no", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)


def status(code):
    return openai.APIStatusError("no", response=httpx.Response(code, request=httpx.Request("POST", "http://x")), body=None)


TRANSPORT = [status(504), status(500), status(403), status(422), openai.APIConnectionError(request=httpx.Request("POST", "http://x"))]
TRANSPORT_IDS = ["504", "500", "403", "422", "connection lost"]


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(meta.time, "sleep", lambda s: None)   # a 500 and a lost connection go through the back-off first


def states(row, ctx):
    return [row["floors"][str(i)]["state"] for i in ctx.stage("floors")["order"]]


def cell(row, ctx, k):
    return row["floors"][str(ctx.stage("floors")["order"][k])]


def sam_record(obj, frame, box, conf=0.9):
    x1, y1, x2, y2 = box
    mask = "data=2,3,!!!!!&^:o6z"
    return f"<{frame}f>{obj}<|box;x1={x1};y1={y1};x2={x2 - 1};y2={y2 - 1};w={FRAME[0]};h={FRAME[1]};c={conf}|><|mask;x=0;y=0;c={conf};{mask}|>\n"


SAM_BOX = (268, 172, 308, 212)   # around PLACE, in stack pixels


# The frozen text. Hashes are of the strings in lab/presence/compare.py, lab/presence/crop.py, lab/tightseed/post1.py and core.py.


@pytest.mark.parametrize("text, digest", [
    (FLOOR_P + EXTRA, "2650eda2a25d98eb"),
    (json.dumps(SCHEMA, sort_keys=True), "7ab50e9a84029456"),
    (QUESTION, "29b7c470edf32946"),
    (json.dumps(ANS, sort_keys=True), "4d440e9ffe4d0cc9"),
    (WHOLE_P, "072adeed7a53f7ba"),
    (json.dumps(GROUND, sort_keys=True), "9b0a4f28687eb9cb"),
], ids=["compare prompt", "compare schema", "crop question", "crop schema", "whole stair prompt", "whole stair schema"])
def test_frozen_text_is_word_for_word(text, digest):
    assert hashlib.sha256(text.encode()).hexdigest()[:16] == digest


# The rule, from the self-checks in lab/presence/rule.py with arm C only


def R(**k):
    return dict(established=True, ink=1.0) | k


def C(**k):
    return dict(v=3, crop="yes", blank=False, aligned=True) | k


@pytest.mark.parametrize("row, c, state", [
    (R(), C(), "found"),
    (R(ink=None), C(), "found"),
    (R(ink=0.1), C(), "check"),
    (R(ink=0.39), C(v=2), "check"),
    (R(ink=0.40), C(v=2), "found"),
    (R(), C(crop="no"), "check"),
    (R(), C(crop="unclear"), "check"),
    (R(), C(v=2, crop="yes"), "found"),
    (R(), C(v=2, crop="no"), "check"),
    (R(), C(v=2, blank=True), "check"),
    (R(ink=0.1), C(v=0, crop="no"), "missing"),
    (R(), C(v=0, crop="yes"), "check"),
    (R(), C(v=0, crop="unclear"), "check"),
    (R(ink=None), C(v=1, blank=True, crop=None), "missing"),
    (R(ink=0.95), C(v=0, crop="no"), "check"),
    (R(ink=0.89), C(v=0, crop="no"), "missing"),
    (R(), C(aligned=False), "check"),
    (R(ink=0.1), C(v=0, crop=None, blank=True, aligned=False), "check"),
    (R(established=False), C(v=1, crop="yes"), "check"),
    (R(established=False), C(v=1, crop=None), "check"),
    (R(established=False), C(v=1, crop="no"), "not_followed"),
    (R(established=False), C(v=0, crop=None), "not_followed"),
    (R(established=False), C(v=1, blank=True), "not_followed"),
])
def test_rule(row, c, state):
    assert decide(row, c)[0] == state


def test_a_failed_crop_never_yields_found():
    state, reason = decide(R(), C(crop=None))
    assert state == "check"
    assert "None" not in reason
    assert decide(R(), C(v=2, crop=None))[0] == "check"


@pytest.mark.parametrize("ink", [None, 0.1, 0.95])
@pytest.mark.parametrize("v", [0, 1])
def test_a_failed_crop_below_two_runs_is_check(v, ink):
    assert decide(R(ink=ink), C(v=v, crop=None)) == ("check", "crop gave no usable reply")


# Clusters, from the self-check in lab/presence/cells.py


def P(r, e, f, x, y):
    return dict(run=r, ent=f"{r}/s/{e}", name="s", floor=f, raw=(x, y), ref=np.array([x, y], float))


POINTS = [P(r, 0, f, 100 + r, 100) for r in range(3) for f in range(2)] + [P(0, 1, 0, 900, 900)] + [P(1, 2, 0, 500, 500), P(1, 2, 1, 700, 500)]


def test_cluster_splits_a_glued_entity():
    assert sorted(len(c["pts"]) for c in cluster(POINTS)) == [1, 1, 1, 6]


def test_points_with_no_place_follow_their_entity():
    u = dict(run=2, ent="2/s/0", name="s", floor=3, raw=(1, 1), ref=None)
    v = dict(run=1, ent="1/s/9", name="s", floor=3, raw=(1, 1), ref=None)
    cs = cluster(POINTS + [u, v])
    assert max(len(c["pts"]) for c in cs) == 7
    assert sum(c["pos"] is None for c in cs) == 1


def test_cluster_radius_is_80():
    assert len(cluster([P(0, 0, 0, 100, 100), P(1, 0, 0, 180, 100)])) == 1
    assert len(cluster([P(0, 0, 0, 100, 100), P(1, 0, 0, 181, 100)])) == 2


def test_a_stray_point_rejoins_its_entitys_row_within_twice_the_radius():
    row = [P(0, 0, 0, 100, 100), P(0, 0, 1, 100, 100)]
    assert len(cluster(row + [P(0, 0, 2, 260, 100)])) == 1
    assert len(cluster(row + [P(0, 0, 2, 261, 100)])) == 2


@pytest.mark.parametrize("name, k, expected", [("F2", 0, 8), (" f3", 0, 9), ("F9", 1, 8), ("Basement", 2, 9), ("Basement", 3, None), ("2", 0, 7)])
def test_floor_index(name, k, expected):
    assert floor_index(name, k, 3, [7, 8, 9]) == expected


# Ink


def test_ink_ratio_is_against_floors_two_runs_called_present(tmp_path):
    paths = {}
    for f, n in enumerate((4, 8, 2, 16)):
        im = np.full((1000, 1000, 3), 255, np.uint8)
        for x in range(n):
            cv2.rectangle(im, (470 + x * 6, 470), (472 + x * 6, 530), (0, 0, 0), -1)
        paths[f] = tmp_path / f"{f}.png"
        cv2.imwrite(str(paths[f]), im)
    row = {"cells": {0: {"v": 3, "chk": (500, 500)}, 1: {"v": 3, "chk": (500, 500)}, 2: {"v": 0, "chk": (500, 500)}, 3: {"v": 1, "chk": None}}}
    ratios, density = ink_ratio(row, paths)
    assert ratios[3] is None and 3 not in density
    assert ratios[0] == pytest.approx(density[0] / density[1], abs=0.01)
    assert ratios[1] == pytest.approx(density[1] / density[0], abs=0.01)
    assert ratios[2] == pytest.approx(density[2] / np.median([density[0], density[1]]), abs=0.01)


def test_ink_ratio_has_no_ratio_with_no_other_claimed_floor(tmp_path):
    path = tmp_path / "0.png"
    cv2.imwrite(str(path), np.full((500, 500, 3), 255, np.uint8))
    assert ink_ratio({"cells": {0: {"v": 3, "chk": (100, 100)}}}, {0: path})[0] == {0: None}


def test_a_place_off_the_sheet_holds_no_ink(tmp_path):
    im = np.full((1000, 1000, 3), 255, np.uint8)
    cv2.rectangle(im, (0, 0), (999, 999), (0, 0, 0), -1)
    path = tmp_path / "0.png"
    cv2.imwrite(str(path), np.pad(im, ((50, 50), (50, 50), (0, 0)), constant_values=255))
    row = {"cells": {0: {"v": 3, "chk": (-70, 500)}, 1: {"v": 3, "chk": (500, 5000)}}}
    assert ink_ratio(row, {0: path, 1: path}, 30)[1] == {0: 0.0, 1: 0.0}


# The stage: comparing floors


def test_three_runs_are_three_model_calls_and_a_rerun_is_free(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(listing(on([0, 1, 2])))
    stage(ctx)
    assert len(m.of("These images")) == 3
    before = len(m.calls)
    stage(ctx)
    assert len(m.calls) == before


def test_fresh_reads_past_the_cache_at_every_spark_call(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(listing(on([0, 1, 2])), sam="")
    questions = "These images", "This is a crop", "You are an object grounding expert"
    stage(ctx)
    assert [len(m.of(q)) for q in questions] == [3, 3, 2]
    ctx.fresh = True
    stage(ctx)
    assert [len(m.of(q)) for q in questions] == [6, 6, 4]


def test_each_run_has_its_own_cache_entry(tmp_path, monkeypatch):
    """Parallel calls on a cold cache all reach the model either way, so the call count alone cannot show this. A rerun with a warm cache would read one reply three times."""
    runs = []

    def spy(prompt, images=(), **kw):
        if prompt.startswith("These images"):
            runs.append(kw.get("run", 0))
        return meta.spark(prompt, images, **kw)
    monkeypatch.setattr(verticals, "spark", spy)
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(listing(on([0, 1, 2])))
    stage(ctx)
    assert sorted(runs) == [0, 1, 2]


def test_compare_request_is_the_lab_request(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()], order=[8, 3, 5], levels=["BASEMENT", "GROUND", "ROOF"])
    m = model(listing(on([0, 1, 2])))
    stage(ctx)
    request = m.of("These images")[0]
    content = request["messages"][-1]["content"]
    assert content[0]["text"] == FLOOR_P.format(order="F1 (BASEMENT), F2 (GROUND), F3 (ROOF)") + EXTRA
    assert request["reasoning_effort"] == "medium"
    assert request["max_completion_tokens"] == 24000
    assert request["response_format"]["json_schema"]["schema"] == SCHEMA
    assert len(content) == 4
    assert all(part["image_url"]["url"].startswith("data:image/jpeg;base64,") for part in content[1:])
    assert max(tile(request).shape) == 2000


def test_floors_sent_in_order_not_sheet_order(tmp_path):
    ctx = review(tmp_path, [stair(), room(), stair(SECOND)], order=[8, 3, 5])
    m = model(listing(on([0])))
    stage(ctx)
    content = m.of("These images")[0]["messages"][-1]["content"]
    assert [p["image_url"]["url"] for p in content[1:]] == [_jpeg_url(ctx.dir / f"floor_{i}.jpg") for i in (8, 3, 5)]


def _jpeg_url(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def test_shaft_claims_are_ignored(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(listing(on([0, 1, 2]), shafts=[on([0, 1, 2], SECOND)]))
    out = stage(ctx)
    assert len(out["verticals"]) == 1
    assert out["verticals"][0]["name"].startswith("Stair")
    assert all("elevator" not in r["messages"][-1]["content"][0]["text"][:40] for r in m.of("This is a crop"))
    assert len(m.of("This is a crop")) == 3
    assert SHAFTS in out["not_checked"]


def test_a_shaft_is_not_a_place_for_a_stair(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(listing(on([0, 1, 2]), shafts=[on([0, 1, 2], (950, 600))]))
    assert cell(stage(ctx)["verticals"][0], ctx, 1)["point"] == pytest.approx(PLACE, abs=2)


def test_row_and_cell_fields(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(listing(on([0, 1, 2])))
    row = stage(ctx)["verticals"][0]
    assert set(row) == {"id", "name", "established", "pos", "floors"}
    assert row["id"] == "A"
    assert row["established"] is True
    assert sorted(row["floors"]) == sorted(str(i) for i in ctx.stage("floors")["order"])
    c = cell(row, ctx, 0)
    assert set(c) == {"state", "signals", "point", "box", "crop", "mask"}
    assert set(c["signals"]) == {"spark", "crop", "ink", "aligned", "reason"}
    assert c["signals"]["spark"] == "3/3"
    assert c["signals"]["aligned"] is True
    assert c["signals"]["crop"] == "yes"
    assert c["signals"]["ink"] == pytest.approx(1.0, abs=0.05)


def test_row_ids_run_past_z(tmp_path):
    places = [(150 + 180 * (n % 9), 150 + 180 * (n // 9)) for n in range(28)]
    ctx = review(tmp_path, [stair(*places), stair(*places)])
    model(listing(*[on([0, 1], p) for p in places], floors=2))
    ids = [row["id"] for row in stage(ctx)["verticals"]]
    assert len(set(ids)) == 28
    assert ids[:2] == ["A", "B"]
    assert ids[25:28] == ["Z", "AA", "AB"]


# The stage: states


def test_ends_become_end_state(tmp_path):
    ctx = review(tmp_path, [room(), stair(), stair(), room()])
    model(listing(on([1, 2]), floors=4))
    row = stage(ctx)["verticals"][0]
    assert states(row, ctx) == ["end", "found", "found", "end"]


def test_missing_between_found_stays_missing(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), room(), stair()])
    model(listing(on([0, 1, 3]), floors=4))
    row = stage(ctx)["verticals"][0]
    assert states(row, ctx) == ["found", "found", "missing", "found"]
    assert cell(row, ctx, 2)["signals"]["reason"]


def test_a_check_floor_counts_in_the_span(tmp_path):
    ctx = review(tmp_path, [stair(), room(), room(), room()])
    model(listing(on([0, 2]), floors=4))   # the runs claim floor 2, where no treads are drawn
    row = stage(ctx)["verticals"][0]
    assert states(row, ctx) == ["found", "missing", "check", "end"]


def test_end_is_by_floor_order_not_sheet_number(tmp_path):
    ctx = review(tmp_path, [stair(), room(), stair(), room()], order=[30, 2, 11, 20])
    model(listing(on([0, 2]), floors=4))
    row = stage(ctx)["verticals"][0]
    assert states(row, ctx) == ["found", "missing", "found", "end"]


def test_one_run_row_stays_visible(tmp_path):
    lone = listing(on([0, 1, 2]), on([1], SECOND))
    plain = listing(on([0, 1, 2]))
    ctx = review(tmp_path, [stair(), stair(PLACE, SECOND), stair()])
    model(lone, plain, plain)
    rows = stage(ctx)["verticals"]
    assert len(rows) == 2
    one = rows[1]
    assert one["established"] is False
    assert states(one, ctx) == ["not_followed", "check", "not_followed"]
    assert cell(one, ctx, 1)["signals"]["spark"] == "1/3"
    assert "one run" in cell(one, ctx, 1)["signals"]["reason"]


def test_one_run_row_the_crop_rejects_is_still_a_row(tmp_path):
    lone = listing(on([0, 1, 2]), on([1], SECOND))
    plain = listing(on([0, 1, 2]))
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(lone, plain, plain)
    one = stage(ctx)["verticals"][1]
    assert one["established"] is False
    assert states(one, ctx) == ["not_followed", "not_followed", "not_followed"]


def test_two_runs_on_one_floor_establish_a_row(tmp_path):
    two = listing(on([1]))
    ctx = review(tmp_path, [room(), stair(), room()])
    model(two, two, listing(on([1], SECOND)))
    rows = stage(ctx)["verticals"]
    assert sorted(r["established"] for r in rows) == [False, True]
    assert states(next(r for r in rows if r["established"]), ctx) == ["end", "found", "end"]


def test_unaligned_floor_is_check(tmp_path):
    ctx = review(tmp_path, [stair(), stair((1500, 600)), stair()], order=[3, 4, 5], reference=3, unaligned=[4])
    model(listing({0: PLACE, 1: (1500, 600), 2: PLACE}))
    row = stage(ctx)["verticals"][0]
    assert states(row, ctx) == ["found", "check", "found"]
    c = cell(row, ctx, 1)
    assert c["signals"]["aligned"] is False
    assert c["signals"]["reason"] == "floor unaligned"
    assert c["point"] == pytest.approx((1500, 600), abs=2)   # no registered place there, so Spark's point


def test_unaligned_floor_marker_is_the_median_of_sparks_points(tmp_path):
    ctx = review(tmp_path, [stair(), stair((1500, 600)), stair()], order=[3, 4, 5], reference=3, unaligned=[4])
    model(listing({0: PLACE, 1: (1500, 600), 2: PLACE}), listing({0: PLACE, 1: (1530, 600), 2: PLACE}), listing({0: PLACE, 1: (1800, 600), 2: PLACE}))
    assert cell(stage(ctx)["verticals"][0], ctx, 1)["point"] == pytest.approx((1530, 600), abs=2)


def test_unaligned_floor_no_run_listed_has_no_point(tmp_path):
    ctx = review(tmp_path, [stair(), blank(), stair()], order=[3, 4, 5], reference=3, unaligned=[4])
    model(listing(on([0, 2])))
    row = stage(ctx)["verticals"][0]
    assert states(row, ctx) == ["found", "check", "found"]
    assert cell(row, ctx, 1)["point"] is None


def test_a_stair_seen_only_on_an_unaligned_floor_is_one_row_per_run(tmp_path):
    ctx = review(tmp_path, [stair(), stair((2400, 1600)), stair()], order=[3, 4, 5], reference=3, unaligned=[4])
    model(listing(on([0, 2]), on([1], (2400, 1600))))
    rows = stage(ctx)["verticals"]
    assert len(rows) == 4
    for row in rows[1:]:
        assert row["pos"] is None
        assert row["established"] is False   # one entity is one run, so such a row can never reach two
        assert states(row, ctx) == ["not_followed", "check", "not_followed"]
        assert cell(row, ctx, 1)["point"] == pytest.approx((2400, 1600), abs=2)


# The stage: pixels


def test_points_are_sheet_pixels(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(listing(on([0, 1, 2])))
    row = stage(ctx)["verticals"][0]
    assert row["pos"] == pytest.approx(PLACE, abs=2)
    assert all(c["point"] == pytest.approx(PLACE, abs=2) for c in row["floors"].values())


def test_a_compare_point_off_the_grid_lands_at_the_sheet_edge(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(listing(on([0, 1, 2], (W * 1.5, -H * 0.15))))   # 1500 and -150 on the 0-1000 grid
    row = stage(ctx)["verticals"][0]
    assert row["pos"] == pytest.approx((W, 0), abs=2)
    assert cell(row, ctx, 1)["point"] == pytest.approx((W, 0), abs=2)


def test_a_transform_in_sheet_pixels_is_read_as_sheet_pixels(tmp_path):
    shifted = (PLACE[0] + 300, PLACE[1] - 150)
    ctx = review(tmp_path, [stair(), stair(shifted), stair()], transforms={1: [[1, 0, -300], [0, 1, 150], [0, 0, 1]]})
    model(listing({0: PLACE, 1: shifted, 2: PLACE}))
    rows = stage(ctx)["verticals"]
    assert len(rows) == 1
    assert states(rows[0], ctx) == ["found", "found", "found"]
    assert cell(rows[0], ctx, 1)["point"] == pytest.approx(shifted, abs=2)
    assert cell(rows[0], ctx, 0)["point"] == pytest.approx(PLACE, abs=2)


def test_sheets_of_different_sizes(tmp_path):
    big = cv2.resize(stair(), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
    ctx = review(tmp_path, [stair(), big, stair()], transforms={1: [[2 / 3, 0, 0], [0, 2 / 3, 0], [0, 0, 1]]})
    model(listing({0: PLACE, 1: PLACE, 2: PLACE}))   # the same place on the 0-1000 grid of a larger sheet
    rows = stage(ctx)["verticals"]
    assert len(rows) == 1
    assert states(rows[0], ctx) == ["found", "found", "found"]
    assert cell(rows[0], ctx, 1)["point"] == pytest.approx((PLACE[0] * 1.5, PLACE[1] * 1.5), abs=3)


def test_pos_is_in_reference_sheet_pixels(tmp_path):
    ctx = review(tmp_path, [stair(), stair((1200, 600)), stair()], order=[3, 4, 5], reference=4, transforms={0: [[1, 0, 300], [0, 1, 0], [0, 0, 1]], 2: [[1, 0, 300], [0, 1, 0], [0, 0, 1]]})
    model(listing({0: PLACE, 1: (1200, 600), 2: PLACE}))
    assert stage(ctx)["verticals"][0]["pos"] == pytest.approx((1200, 600), abs=2)


def test_the_marker_is_the_registered_place_not_sparks_point(tmp_path):
    drifted = (PLACE[0] + 90, PLACE[1])   # 60 px on the 2000 px copy, inside the cluster radius
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(listing({0: PLACE, 1: drifted, 2: PLACE}))
    rows = stage(ctx)["verticals"]
    assert len(rows) == 1
    assert cell(rows[0], ctx, 1)["point"] == pytest.approx(PLACE, abs=2)


# The stage: crops


def test_crop_is_saved_and_named(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()], order=[3, 4, 5])
    model(listing(on([0, 1, 2])))
    row = stage(ctx)["verticals"][0]
    c = row["floors"]["4"]
    assert c["crop"] == "verticals/A_4.jpg"
    saved = cv2.imread(str(tmp_path / c["crop"]))
    assert saved.shape == (780, 780, 3)


def test_crop_request_is_the_lab_request(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(listing(on([0, 1, 2])))
    stage(ctx)
    request = m.of("This is a crop")[0]
    content = request["messages"][-1]["content"]
    assert content[0]["text"] == QUESTION
    assert request["reasoning_effort"] == "low"
    assert request["max_completion_tokens"] == 4000
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert tile(request).shape == (780, 780)


def test_crop_centre_is_snapped_to_30_px(tmp_path, monkeypatch):
    seen = []
    cut = verticals._cut
    monkeypatch.setattr(verticals, "_cut", lambda path, x, y: seen.append((x, y)) or cut(path, x, y))
    ctx = review(tmp_path, [stair((904, 616)), stair((904, 616))])
    model(listing(on([0, 1], (904, 616)), floors=2))   # 603 and 411 on the 2000 px copy
    stage(ctx)
    assert set(seen) == {(600, 420)}


def test_a_crop_centre_off_the_sheet_still_cuts(tmp_path):
    ctx = review(tmp_path, [stair((90, 90)), stair((0, 90)), stair((90, 90))], transforms={1: [[1, 0, 120], [0, 1, 0], [0, 0, 1]]})
    model(listing(on([0, 2], (90, 90))))
    c = cell(stage(ctx)["verticals"][0], ctx, 1)
    assert c["point"][0] == pytest.approx(-30, abs=2)   # ink at the sheet's edge keeps this place from reading blank, so it gets a crop
    assert cv2.imread(str(tmp_path / c["crop"])).shape == (780, 780, 3)


def test_ink_is_read_in_the_60_px_box(tmp_path):
    ctx = review(tmp_path, [stair(), room(), stair()])
    model(listing(on([0, 1, 2])), crop={"answer": "yes", "seen": "x"})   # the walls sit in a 100 px box and outside a 60 px box
    c = cell(stage(ctx)["verticals"][0], ctx, 1)
    assert c["signals"]["ink"] == 0
    assert (c["state"], c["signals"]["reason"]) == ("check", "ink gone at the registered place")


def test_blank_place_gets_no_crop(tmp_path):
    ctx = review(tmp_path, [stair(), blank(), stair()])
    m = model(listing(on([0, 2])))
    row = stage(ctx)["verticals"][0]
    assert len(m.of("This is a crop")) == 2
    c = cell(row, ctx, 1)
    assert c["crop"] is None and c["signals"]["crop"] is None
    assert states(row, ctx) == ["found", "missing", "found"]


def test_every_cell_of_an_established_row_is_cropped(tmp_path):
    ctx = review(tmp_path, [stair(), room(), stair(), room()])
    m = model(listing(on([0, 2]), floors=4))
    stage(ctx)
    assert len(m.of("This is a crop")) == 4


def test_other_rows_are_cropped_only_where_a_run_saw_them(tmp_path):
    lone = listing(on([0, 1, 2]), on([1], SECOND))
    plain = listing(on([0, 1, 2]))
    ctx = review(tmp_path, [stair(PLACE, SECOND), stair(PLACE, SECOND), stair(PLACE, SECOND)])
    m = model(lone, plain, plain)
    one = stage(ctx)["verticals"][1]
    assert len(m.of("This is a crop")) == 3 + 1
    assert cell(one, ctx, 0)["crop"] is None
    assert cell(one, ctx, 1)["crop"] is not None


@pytest.mark.parametrize("reply", ["", "not json", chat("", finish="content_filter"), chat('{"answer": "yes", "seen": "x"}', finish="length"), {"answer": "maybe", "seen": "x"}, {"seen": "x"}, ["yes"], refused()],
                         ids=["empty", "malformed", "filtered", "cut off", "answer not in the list", "no answer", "wrong shape", "refused"])
def test_empty_crop_reply_is_check(tmp_path, reply):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(listing(on([0, 1, 2])), crop=reply)
    out = stage(ctx)
    row = out["verticals"][0]
    assert states(row, ctx) == ["check", "check", "check"]
    c = cell(row, ctx, 0)
    assert c["signals"]["crop"] is None
    assert c["signals"]["reason"]
    assert c["crop"] == "verticals/A_" + str(ctx.stage("floors")["order"][0]) + ".jpg"
    assert any("crop" in line for line in out["not_checked"])


def test_one_failed_crop_leaves_the_others(tmp_path):
    ctx = review(tmp_path, [stair(), disc_on(stair()), stair()])
    model(listing(on([0, 1, 2])), crop=no_reply_on_the_disc)
    out = stage(ctx)
    assert states(out["verticals"][0], ctx) == ["found", "check", "found"]
    assert "Stair crops with no usable reply: 1. A cell with no crop reply shows as Check." in out["not_checked"]


def test_a_failed_crop_at_one_run_past_the_found_floors_is_check(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair(), disc_on(thin_stair())])
    rest = listing(on([0, 1, 2]), floors=4)
    model(listing(on([0, 1, 2, 3]), floors=4), rest, rest, crop=no_reply_on_the_disc)
    out = stage(ctx)
    row = out["verticals"][0]
    c = cell(row, ctx, 3)
    assert states(row, ctx) == ["found", "found", "found", "check"]
    assert (c["signals"]["spark"], c["signals"]["reason"]) == ("1/3", "crop gave no usable reply")
    assert "Stair crops with no usable reply: 1. A cell with no crop reply shows as Check." in out["not_checked"]


def test_a_failed_crop_no_run_listed_between_found_floors_is_check(tmp_path):
    ctx = review(tmp_path, [stair(), disc_on(thin_stair()), stair()])
    model(listing(on([0, 2])), crop=no_reply_on_the_disc)
    assert states(stage(ctx)["verticals"][0], ctx) == ["found", "check", "found"]


@pytest.mark.parametrize("failure", TRANSPORT, ids=TRANSPORT_IDS)
def test_a_transport_failure_on_one_crop_costs_that_crop(tmp_path, failure, no_wait):
    def fails_on_the_disc(request):
        if has_disc(request):
            raise failure
        return looks(request)
    ctx = review(tmp_path, [stair(), disc_on(stair()), stair()])
    model(listing(on([0, 1, 2])), crop=fails_on_the_disc)
    out = stage(ctx)
    assert states(out["verticals"][0], ctx) == ["found", "check", "found"]
    assert any(line.startswith("Stair crops with no usable reply: 1.") for line in out["not_checked"])


def test_budget_during_crops_fails_the_stage_for_the_pipeline(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()], cap=0.0002)
    model(listing(on([0, 1, 2])))
    with pytest.raises(BudgetReached):
        stage(ctx)


# The stage: comparison failures


@pytest.mark.parametrize("reply", ["", "not json", {"stairs": [], "shafts": []}, {"shafts": []}, {"stairs": "none", "shafts": []}, {"stairs": [{"name": "x", "floors": [{"floor": "F1"}]}], "shafts": []},
                                   {"stairs": [{"name": "x", "floors": [{"floor": "F1", "present": True, "x": "12", "y": 3}]}], "shafts": []},
                                   {"stairs": [{"name": 7, "floors": [{"floor": "F1", "present": True, "x": 12, "y": 3}]}], "shafts": []}, ["stairs"], chat("", finish="length"), refused(),
                                   '{"stairs": [{"name": "x", "floors": [{"floor": "F1", "present": true, "x": NaN, "y": 3}]}], "shafts": []}'],
                         ids=["empty", "malformed", "blank lists", "no stairs key", "stairs not a list", "no point", "text coordinate", "name not text", "wrong shape", "cut off", "refused", "not a number"])
def test_empty_compare_reply_gives_not_checked(tmp_path, reply):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(reply)
    out = stage(ctx)
    assert out["verticals"] == []
    assert len(out["not_checked"]) == 2
    assert SHAFTS in out["not_checked"]
    assert any("placed a stair" in line for line in out["not_checked"])   # a set with no stairs lands here too, because a blank answer arrives the same way
    assert not m.of("This is a crop")
    assert not m.sam_prompts


def test_one_failed_run_scores_what_is_left(tmp_path):
    good = listing(on([0, 1, 2]))
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(good, good, "", "")   # the third run and its one retry
    out = stage(ctx)
    row = out["verticals"][0]
    assert row["established"] is True
    assert states(row, ctx) == ["found", "found", "found"]
    assert cell(row, ctx, 0)["signals"]["spark"] == "2/3"
    assert any("1 of the 3" in line for line in out["not_checked"])


@pytest.mark.parametrize("failure", TRANSPORT, ids=TRANSPORT_IDS)
def test_a_transport_failure_on_one_run_scores_what_is_left(tmp_path, failure, no_wait):
    good = listing(on([0, 1, 2]))
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(good, good, failure)
    out = stage(ctx)
    row = out["verticals"][0]
    assert states(row, ctx) == ["found", "found", "found"]
    assert cell(row, ctx, 0)["signals"]["spark"] == "2/3"
    assert any("1 of the 3" in line for line in out["not_checked"])


@pytest.mark.parametrize("failure", [status(504), status(500), TRANSPORT[-1]], ids=["504", "500", "connection lost"])
def test_no_run_read_after_transport_failures_fails_the_stage_so_a_rerun_retries(tmp_path, failure, no_wait):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(failure)
    with pytest.raises(openai.APIError):
        stage(ctx)
    assert not m.of("This is a crop")


def test_one_transport_failure_among_bad_replies_also_fails_the_stage(tmp_path, monkeypatch):
    def last_run_gets_a_504(prompt, images=(), run=0, **kw):
        if run == 2:
            raise status(504)
        return {"stairs": [], "shafts": []}
    monkeypatch.setattr(verticals, "spark", last_run_gets_a_504)
    with pytest.raises(openai.APIError):
        stage(review(tmp_path, [stair(), stair(), stair()]))


def test_two_failed_runs_leave_nothing_established(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(listing(on([0, 1, 2])), "", "", "", "")
    out = stage(ctx)
    row = out["verticals"][0]
    assert row["established"] is False
    assert states(row, ctx) == ["check", "check", "check"]
    assert any("2 of the 3" in line for line in out["not_checked"])


def test_a_run_outside_the_floor_codes_falls_back_to_position(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    named = {"stairs": [{"name": "s", "floors": [{"floor": lab, "present": True, "x": 300, "y": 300} for lab in ("BASEMENT", "GROUND", "ROOF")]}], "shafts": []}
    model(named)
    assert states(stage(ctx)["verticals"][0], ctx) == ["found", "found", "found"]


def test_budget_during_compare_fails_the_stage_for_the_pipeline(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    ctx.meter.add(ctx.meter.cap_usd)
    model(listing(on([0, 1, 2])))
    with pytest.raises(BudgetReached):
        stage(ctx)


# The stage: no stack


@pytest.mark.parametrize("floors", [
    {"order": [4], "reference": 4, "transforms": {}, "inliers": {}, "unaligned": [], "stack": None, "frame_size": None, "discharge": None, "not_checked": "Stairs were not followed across floors, because this set has one floor plan."},
    {"order": [], "reference": None, "transforms": {}, "inliers": {}, "unaligned": [], "stack": None, "frame_size": None, "discharge": None, "not_checked": "Stairs were not followed across floors, because this set has no floor plans."},
    {"failed": "x"},
    {},
], ids=["one floor", "no floors", "failed", "missing"])
def test_too_few_floors_gives_not_checked_and_no_call(tmp_path, floors):
    (tmp_path / "stage_read.json").write_text(json.dumps({"sheets": []}))
    (tmp_path / "stage_floors.json").write_text(json.dumps(floors))
    ctx = Context(tmp_path, Meter(ledger=tmp_path / "ledger.jsonl"), [], False)
    meta.set_transport(lambda kind, **request: pytest.fail("no model call"))
    out = stage(ctx)
    assert out["verticals"] == []
    assert SHAFTS in out["not_checked"]
    assert any(line != SHAFTS and "not followed" in line for line in out["not_checked"])
    if floors.get("not_checked"):
        assert floors["not_checked"] in out["not_checked"]


# The stage: masks


def test_mask_only_on_found_cells(tmp_path):
    ctx = review(tmp_path, [stair(), room(), room(), stair()])
    records = "".join(sam_record(1, f, SAM_BOX) for f in range(4))
    m = model(listing(on([0, 1, 3]), floors=4), sam=records)
    row = stage(ctx)["verticals"][0]
    assert states(row, ctx) == ["found", "check", "missing", "found"]
    assert cell(row, ctx, 0)["mask"] and cell(row, ctx, 3)["mask"]
    assert cell(row, ctx, 1)["mask"] is None and cell(row, ctx, 2)["mask"] is None
    assert cell(row, ctx, 1)["box"] is None
    assert len(m.sam_prompts) == 1


def test_sam_is_pinned_on_every_found_frame_and_nowhere_else(tmp_path):
    ctx = review(tmp_path, [stair(), room(), room(), stair()])
    m = model(listing(on([0, 1, 3]), floors=4), sam="")
    stage(ctx)
    prompt = m.sam_prompts[0]
    assert prompt.count("<|box;") == 2
    assert prompt.startswith("<0f>1<|box;") and "<3f>1<|box;" in prompt


def test_seed_is_the_whole_stair_box_in_stack_pixels(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(listing(on([0, 1, 2])), sam="")
    stage(ctx)
    assert len(m.of("You are an object grounding expert")) == 2
    # Round one: a 300 px window at 600, 400 on the copy, the answer's middle fifth is 570-630 by 369-429. Round two: a 200 px window, 580-620 by 379-419. A frame is 0.48 of the 2000 by 1333 copy.
    assert m.sam_prompts[0].startswith("<0f>1<|box;x1=278;y1=182;x2=298;y2=201;w=960;h=640|>")


def test_each_seed_crop_is_enlarged_to_1000_px(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(listing(on([0, 1, 2])), sam="")
    stage(ctx)
    assert [tile(r).shape for r in m.of("You are an object grounding expert")] == [(1000, 1000)] * 2


def test_second_seed_window_is_1_8_times_the_first_box(tmp_path):
    answers = [{"objects": [{"object_name": "stair", "x_min": 200, "y_min": 200, "x_max": 800, "y_max": 800}]}, WHOLE]
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(listing(on([0, 1, 2])), whole=lambda r: answers.pop(0), sam="")
    stage(ctx)
    # Round one: the middle 60% of a 300 px window is a 180 px box. Round two: a 324 px window, whose middle fifth is 567.6-632.4 by 366.6-431.4 on the copy.
    assert m.sam_prompts[0].startswith("<0f>1<|box;x1=272;y1=176;x2=304;y2=207;")


def test_seed_box_moves_through_the_reference_frame(tmp_path):
    shifted = (PLACE[0] + 300, PLACE[1])
    ctx = review(tmp_path, [room(), stair(shifted), room()], transforms={1: [[1, 0, -300], [0, 1, 0], [0, 0, 1]]}, order=[3, 4, 5], reference=3)
    m = model(listing({1: shifted}), sam="")
    stage(ctx)
    assert m.sam_prompts[0] == "<1f>1<|box;x1=278;y1=182;x2=298;y2=201;w=960;h=640|>"


def test_seed_answer_nearest_the_place_wins(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    far = {"object_name": "stair", "x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100}
    near = WHOLE["objects"][0]
    m = model(listing(on([0, 1, 2])), whole={"objects": [far, near]}, sam="")
    stage(ctx)
    assert m.sam_prompts[0].startswith("<0f>1<|box;x1=278;y1=182;x2=298;y2=201;")


def test_seed_comes_from_the_median_found_floor(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(verticals, "_whole", lambda ctx, path, place: seen.append(path.name) or None)
    ctx = review(tmp_path, [stair(), room(), stair(), stair(), stair()], order=[1, 2, 3, 4, 5])
    model(listing(on([0, 2, 3, 4]), floors=5))
    stage(ctx)
    assert seen == ["floor_3.jpg"]   # positions 0, 2, 3, 4, and the lower middle one on a tie


def test_no_seed_box_means_no_sam_object(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(listing(on([0, 1, 2])), whole={"objects": []})
    out = stage(ctx)
    assert not m.sam_prompts
    assert all(c["mask"] is None for c in out["verticals"][0]["floors"].values())
    assert states(out["verticals"][0], ctx) == ["found", "found", "found"]


def test_second_round_empty_keeps_the_first_box(tmp_path):
    answers = [WHOLE, {"objects": []}]
    ctx = review(tmp_path, [stair(), room(), room()])
    m = model(listing(on([0])), whole=lambda r: answers.pop(0), sam="")
    stage(ctx)
    assert m.sam_prompts[0] == "<0f>1<|box;x1=274;y1=177;x2=302;y2=206;w=960;h=640|>"


def test_a_row_with_no_found_cell_costs_no_seed(tmp_path):
    ctx = review(tmp_path, [room(), room(), room()])
    m = model(listing(on([0, 1, 2])))
    stage(ctx)
    assert not m.of("You are an object grounding expert")
    assert not m.sam_prompts


def test_mask_is_a_polygon_in_that_floors_sheet_pixels(tmp_path):
    shifted = (PLACE[0] + 300, PLACE[1] - 150)
    ctx = review(tmp_path, [stair(), stair(shifted), stair()], transforms={1: [[1, 0, -300], [0, 1, 150], [0, 0, 1]]})
    model(listing({0: PLACE, 1: shifted, 2: PLACE}), sam="".join(sam_record(1, f, SAM_BOX) for f in range(3)))
    row = stage(ctx)["verticals"][0]
    for k, (dx, dy) in {0: (0, 0), 1: (300, -150)}.items():
        c = cell(row, ctx, k)
        xs, ys = zip(*c["mask"])
        assert len(c["mask"]) >= 3
        assert c["box"] == [min(xs), min(ys), max(xs), max(ys)]
        # The stack box is 268-308 by 172-212, which is 837.5-962.5 by 537.5-662.5 in sheet pixels.
        assert c["box"][0] == pytest.approx(837.5 + dx, abs=4)
        assert c["box"][1] == pytest.approx(537.5 + dy, abs=4)
        assert c["box"][2] == pytest.approx(962.5 + dx, abs=5)
        assert c["box"][3] == pytest.approx(662.5 + dy, abs=5)


def test_mask_under_half_confidence_is_dropped(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(listing(on([0, 1, 2])), sam=sam_record(1, 0, SAM_BOX, conf=0.49) + sam_record(1, 1, SAM_BOX, conf=0.5))
    row = stage(ctx)["verticals"][0]
    assert cell(row, ctx, 0)["mask"] is None
    assert cell(row, ctx, 1)["mask"] is not None


@pytest.mark.parametrize("bad", [sam_record(1, 0, SAM_BOX).replace(";c=0.9", ""), sam_record(1, 9, SAM_BOX), sam_record(7, 0, SAM_BOX)], ids=["no confidence", "frame past the stack", "object nobody asked for"])
def test_a_bad_mask_record_is_skipped_and_the_next_one_kept(tmp_path, bad):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    model(listing(on([0, 1, 2])), sam=bad + sam_record(1, 1, SAM_BOX))
    row = stage(ctx)["verticals"][0]
    assert cell(row, ctx, 0)["mask"] is None
    assert cell(row, ctx, 1)["mask"] is not None


def test_mask_past_120_px_of_the_place_is_dropped(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    x1, y1, x2, y2 = SAM_BOX
    edge = (PLACE[0] * 0.32 + 120 * 0.48)   # the place in stack pixels, plus 120 px on the 2000 px copy
    near = (int(edge) - 2, y1, int(edge) + 40, y2)
    far = (int(edge) + 3, y1, int(edge) + 40, y2)
    model(listing(on([0, 1, 2])), sam=sam_record(1, 0, near) + sam_record(1, 1, far))
    row = stage(ctx)["verticals"][0]
    assert cell(row, ctx, 0)["mask"] is not None
    assert cell(row, ctx, 1)["mask"] is None


@pytest.mark.parametrize("flip", [False, True], ids=["large blob below", "large blob above"])
def test_outline_is_the_largest_blob(flip):
    mask = np.zeros((40, 60), bool)
    mask[2:6, 2:6] = True
    mask[10:35, 20:55] = True
    outline = verticals._outline(mask[::-1] if flip else mask, 100, 200)
    top, bottom = (205, 229) if flip else (210, 234)
    assert (outline[:, 0].min(), outline[:, 0].max()) == (120, 154)
    assert (outline[:, 1].min(), outline[:, 1].max()) == (top, bottom)


@pytest.mark.parametrize("failure", [RuntimeError("SAM response.failed"), "not a segmentation stream", BudgetReached("cap")], ids=["stream failed", "unreadable", "budget"])
def test_sam_failure_changes_no_state(tmp_path, failure):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    run = listing(on([0, 2]))
    ctx = review(tmp_path / "a", [stair(), room(), stair()])
    model(run, sam="")
    clean = stage(ctx)
    meta.CACHE = tmp_path / "other cache"
    ctx = review(tmp_path / "b", [stair(), room(), stair()])
    model(run, sam=failure)
    out = stage(ctx)
    assert out["verticals"] == clean["verticals"]
    assert any("outline" in line for line in out["not_checked"])


def test_seed_failure_changes_no_state(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    m = model(listing(on([0, 1, 2])), whole="")
    out = stage(ctx)
    assert states(out["verticals"][0], ctx) == ["found", "found", "found"]
    assert not m.sam_prompts
    assert any("outline" in line for line in out["not_checked"])
    assert not any("spend cap" in line for line in out["not_checked"])


def test_budget_at_the_seeds_keeps_the_matrix(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()], cap=0.0021)   # six calls at 0.00034 are the runs and crops, and the seventh, the first seed crop, is the last the cap lets through
    m = model(listing(on([0, 1, 2])))
    out = stage(ctx)
    assert states(out["verticals"][0], ctx) == ["found", "found", "found"]
    assert len(m.of("You are an object grounding expert")) == 1
    assert not m.sam_prompts
    assert any("spend cap" in line and "outline" in line for line in out["not_checked"])


def test_budget_at_sam_keeps_the_matrix(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()], cap=0.0027)   # eight calls at 0.00034 are the runs, crops and seed rounds, and the cap stops the SAM call after them
    m = model(listing(on([0, 1, 2])))
    out = stage(ctx)
    assert states(out["verticals"][0], ctx) == ["found", "found", "found"]
    assert len(m.of("You are an object grounding expert")) == 2
    assert not m.sam_prompts
    assert any("spend cap" in line and "outline" in line for line in out["not_checked"])


def test_a_missing_stack_leaves_masks_null(tmp_path):
    ctx = review(tmp_path, [stair(), stair(), stair()])
    (tmp_path / "stack.mp4").unlink()
    model(listing(on([0, 1, 2])))
    out = stage(ctx)
    assert states(out["verticals"][0], ctx) == ["found", "found", "found"]
    assert all(c["mask"] is None for c in out["verticals"][0]["floors"].values())


def test_seventeen_stairs_split_into_two_sam_requests(tmp_path):
    places = [(150 + 300 * (n % 9), 150 + 300 * (n // 9)) for n in range(17)]
    ctx = review(tmp_path, [stair(*places), stair(*places)], frame=FRAME)
    m = model(listing(*[on([0, 1], p) for p in places], floors=2), sam="")
    stage(ctx)
    prompts = m.sam_prompts
    assert len(prompts) == 2
    objects = [{int(x.split("<|box")[0].split(">")[-1].lstrip(",")) for x in p.split("|>")[:-1]} for p in prompts]
    assert [len(o) for o in objects] == [16, 1]
    assert not objects[0] & objects[1]


def test_one_failed_sam_request_keeps_the_other_requests_masks(tmp_path):
    places = [(150 + 300 * (n % 9), 150 + 300 * (n // 9)) for n in range(17)]
    ctx = review(tmp_path, [stair(*places), stair(*places)], frame=FRAME)

    class Flaky(Model):
        def __call__(self, kind, **request):
            if kind == "sam" and sum(k == "sam" for k, _ in self.calls) == 0:
                self.calls.append((kind, request))
                raise RuntimeError("SAM response.failed")
            return super().__call__(kind, **request)
    x, y = round(places[16][0] * 0.32), round(places[16][1] * 0.32)
    m = Flaky(listing(*[on([0, 1], p) for p in places], floors=2), sam=sam_record(17, 0, (x - 20, y - 20, x + 20, y + 20)))
    meta.set_transport(m)
    rows = stage(ctx)["verticals"]
    assert cell(rows[16], ctx, 0)["mask"] is not None
    assert cell(rows[0], ctx, 0)["mask"] is None
