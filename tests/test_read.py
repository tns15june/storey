import base64
import json
import threading
import time

import cv2
import httpx
import numpy as np
import openai
import pytest

from storey import meta
from storey.meta import BudgetReached, Meter
from storey.pipeline import Context
from storey.read import read_stage

GOOD = {"sheet_class": "floor_plan", "sheet_number": "A2.1", "sheet_title": "MAIN FLOOR PLAN", "level": "GROUND FLOOR", "scale": "1/4\" = 1'-0\"",
        "plan_box": {"x1": 100, "y1": 200, "x2": 900, "y2": 800}}
BLANK = {"sheet_class": "other", "sheet_number": "", "sheet_title": "", "level": "", "scale": "", "plan_box": {"x1": 0, "y1": 0, "x2": 0, "y2": 0}}
REPLY = {"text": json.dumps(GOOD), "input": 1000, "cached": 0, "output": 100, "finish": "stop"}
UNREAD = {"number": "", "title": "", "cls": "other", "level": "", "scale": "", "read_ok": False}


def reply(**fields):
    return {**REPLY, "text": json.dumps(GOOD | fields)}


def text(obj):
    return {"text": json.dumps(obj)}


# A reply that spark itself rejects, then replies it accepts as JSON that break the schema, then the blank form an "empty" answer takes under constrained decoding.
BAD = {
    "none": {"text": None},
    "whitespace": {"text": "  \n"},
    "not-json": {"text": "not json"},
    "not-an-object": {"text": "[1, 2]"},
    "cut-off": {"text": '{"sheet_class"', "finish": "length"},
    "filtered": {"text": None, "finish": "content_filter"},
    "no-fields": text({}),
    "missing-box": text({k: v for k, v in GOOD.items() if k != "plan_box"}),
    "unknown-class": text(GOOD | {"sheet_class": "approved"}),
    "number-not-text": text(GOOD | {"sheet_number": 7}),
    "box-not-object": text(GOOD | {"plan_box": [1, 2, 3, 4]}),
    "box-not-integers": text(GOOD | {"plan_box": {"x1": "a", "y1": 2.5, "x2": None, "y2": True}}),
    "box-float": text(GOOD | {"plan_box": GOOD["plan_box"] | {"x1": 100.5}}),
    "box-bool": text(GOOD | {"plan_box": GOOD["plan_box"] | {"x1": True}}),
    "blank-fields": text(BLANK),
}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("a test reached the real client"))
    yield
    meta.set_transport(None)


def make(tmp_path, sizes, cap_usd=8.0):
    """A Context with one solid sheet per (width, height). Each sheet's grey is its index, so a fake can tell them apart."""
    sheets = []
    for i, (w, h) in enumerate(sizes):
        name = f"sheet_{i:03d}.png"
        cv2.imwrite(str(tmp_path / name), np.full((h, w, 3), i, np.uint8))
        sheets.append({"index": i, "image": name, "width": w, "height": h, "kind": "scan", "text": "", "words": []})
    return Context(tmp_path, Meter(cap_usd=cap_usd, ledger=tmp_path / "ledger.jsonl"), sheets, False)


def sent_image(request):
    url = request["messages"][0]["content"][1]["image_url"]["url"]
    return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)


def sheet_of(request):
    return int(sent_image(request)[0, 0, 0])


def always(answer):
    calls = []
    meta.set_transport(lambda kind, **request: calls.append(request) or answer)
    return calls


def api_error(cls, code):
    return cls("no", response=httpx.Response(code, request=httpx.Request("POST", "http://x")), body=None)


def test_prompt_uses_2000px_image(tmp_path):
    calls = always(REPLY)
    read_stage(make(tmp_path, [(4000, 3000)]))
    assert sent_image(calls[0]).shape == (1500, 2000, 3)
    assert len(calls[0]["messages"][0]["content"]) == 2
    assert (calls[0]["model"], calls[0]["reasoning_effort"]) == ("muse-spark-1.3", "low")
    assert calls[0]["max_completion_tokens"] == 24000
    assert calls[0]["response_format"]["json_schema"]["strict"] is True


def test_tall_sheet_is_limited_by_its_long_edge(tmp_path):
    calls = always(REPLY)
    read_stage(make(tmp_path, [(3000, 6000)]))
    assert sent_image(calls[0]).shape == (2000, 1000, 3)


def test_small_sheet_is_not_upscaled(tmp_path):
    calls = always(REPLY)
    read_stage(make(tmp_path, [(800, 600)]))
    assert sent_image(calls[0]).shape == (600, 800, 3)


def test_grid_coordinates_map_to_pixels(tmp_path):
    always(reply(plan_box={"x1": 500, "y1": 250, "x2": 750, "y2": 1000}))
    assert read_stage(make(tmp_path, [(4000, 3000)]))["sheets"][0]["plan_box"] == [2000, 750, 3000, 3000]


@pytest.mark.parametrize("box, want", [
    ({"x1": 0, "y1": 0, "x2": 0, "y2": 0}, [0, 0, 4000, 3000]),
    ({"x1": 900, "y1": 100, "x2": 100, "y2": 800}, [0, 0, 4000, 3000]),
    ({"x1": 100, "y1": 500, "x2": 900, "y2": 500}, [0, 0, 4000, 3000]),
    ({"x1": -50, "y1": 100, "x2": 1200, "y2": 800}, [0, 300, 4000, 2400]),
    ({"x1": 2000, "y1": 2000, "x2": 3000, "y2": 3000}, [0, 0, 4000, 3000]),
], ids=["no-plan", "swapped", "flat", "past-the-edges", "off-the-sheet"])
def test_plan_box_is_always_a_usable_box_on_the_sheet(tmp_path, box, want):
    always(reply(plan_box=box))
    row = read_stage(make(tmp_path, [(4000, 3000)]))["sheets"][0]
    assert row["plan_box"] == want
    assert row["read_ok"] is True


def test_row_carries_each_field_of_the_interface(tmp_path):
    always(reply(sheet_number=" A2.1\n", sheet_title=" MAIN FLOOR PLAN "))
    assert read_stage(make(tmp_path, [(1000, 500)])) == {"sheets": [{
        "index": 0, "number": "A2.1", "title": "MAIN FLOOR PLAN", "cls": "floor_plan", "level": "GROUND FLOOR", "scale": "1/4\" = 1'-0\"",
        "plan_box": [100, 100, 900, 400], "read_ok": True,
    }]}


@pytest.mark.parametrize("bad", BAD.values(), ids=BAD)
def test_empty_or_bad_reply_marks_sheet_unread(tmp_path, bad):
    meta.set_transport(lambda kind, **request: {**REPLY, **bad} if sheet_of(request) == 1 else REPLY)
    rows = read_stage(make(tmp_path, [(40, 30), (40, 30), (40, 30)]))["sheets"]
    assert rows[1] == UNREAD | {"index": 1, "plan_box": [0, 0, 40, 30]}
    assert [r["read_ok"] for r in rows] == [True, False, True]
    assert rows[0]["number"] == rows[2]["number"] == "A2.1"


@pytest.mark.parametrize("fields", [
    BLANK | {"sheet_class": "floor_plan"},
    BLANK | {"sheet_title": "GENERAL NOTES"},
    BLANK | {"level": "BASEMENT"},
], ids=["class", "title", "level"])
def test_sheet_with_no_printed_number_still_counts_as_read(tmp_path, fields):
    always(reply(**fields))
    assert read_stage(make(tmp_path, [(40, 30)]))["sheets"][0]["read_ok"] is True


def test_sheet_image_that_cannot_be_opened_is_unread_and_costs_nothing(tmp_path):
    ctx = make(tmp_path, [(40, 30), (40, 30)])
    (tmp_path / "sheet_000.png").write_bytes(b"not a png")
    calls = always(REPLY)
    rows = read_stage(ctx)["sheets"]
    assert rows[0] == UNREAD | {"index": 0, "plan_box": [0, 0, 40, 30]}
    assert rows[1]["read_ok"] is True
    assert len(calls) == 1


def test_parallel_calls_keep_sheet_order(tmp_path):
    ctx = make(tmp_path, [(40, 30)] * 16)
    barrier, lock = threading.Barrier(8), threading.Lock()   # a call waits here until eight are running, so a run with fewer workers fails
    running, peak = [], []

    def transport(kind, **request):
        i = sheet_of(request)
        with lock:
            running.append(i)
            peak.append(len(running))
        barrier.wait(10)
        time.sleep((8 - i % 8) * 0.01)   # later sheets in each group of eight finish first
        with lock:
            running.remove(i)
        return reply(sheet_number=f"N{i}")
    meta.set_transport(transport)
    rows = read_stage(ctx)["sheets"]
    assert [r["index"] for r in rows] == list(range(16))
    assert [r["number"] for r in rows] == [f"N{i}" for i in range(16)]
    assert max(peak) == 8


def test_sheet_the_api_refuses_is_unread_and_the_others_proceed(tmp_path):
    calls = []

    def transport(kind, **request):
        calls.append(request)
        if sheet_of(request) == 1:
            raise api_error(openai.BadRequestError, 400)
        return REPLY
    meta.set_transport(transport)
    rows = read_stage(make(tmp_path, [(40, 30)] * 3))["sheets"]
    assert rows[1] == UNREAD | {"index": 1, "plan_box": [0, 0, 40, 30]}
    assert [r["read_ok"] for r in rows] == [True, False, True]
    assert len(calls) == 3


@pytest.mark.parametrize("code", [401, 402, 403, 404, 429, 500, 504])
def test_api_error_that_is_not_about_one_sheet_fails_the_stage(tmp_path, monkeypatch, code):
    monkeypatch.setattr(meta.time, "sleep", lambda s: None)

    def transport(kind, **request):
        raise api_error(openai.APIStatusError, code)
    meta.set_transport(transport)
    with pytest.raises(openai.APIStatusError):
        read_stage(make(tmp_path, [(40, 30)]))


def test_budget_stop_fails_the_stage_instead_of_marking_sheets_unread(tmp_path):
    calls = always(REPLY)
    with pytest.raises(BudgetReached):
        read_stage(make(tmp_path, [(40, 30), (40, 30)], cap_usd=0))
    assert calls == []


def test_second_run_comes_from_the_cache_unless_fresh(tmp_path):
    ctx = make(tmp_path, [(4000, 3000), (4000, 3000)])
    calls = always(REPLY)
    read_stage(ctx)
    spent = ctx.meter.spent
    read_stage(ctx)
    assert len(calls) == 2
    assert ctx.meter.spent == spent
    ctx.fresh = True
    read_stage(ctx)
    assert len(calls) == 4


def test_progress_counts_the_sheets(tmp_path):
    ctx = make(tmp_path, [(40, 30)] * 3)
    ctx.status = {"stages": [{"state": "running"}]}
    always(REPLY)
    read_stage(ctx)
    assert ctx.status["stages"] == [{"state": "running", "done": 3, "total": 3}]
    assert ctx.status["step"] == "Reading sheet 3 of 3"
