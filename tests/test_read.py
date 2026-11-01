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
from storey.findings import evaluate
from storey.meta import SAM_IMAGE_USD, BudgetReached, Meter
from storey.pipeline import Context
from storey.read import CLASSES, read_stage

GOOD = {"sheet_class": "floor_plan", "sheet_number": "A2.1", "sheet_title": "MAIN FLOOR PLAN", "level": "GROUND FLOOR", "scale": "1/4\" = 1'-0\"",
        "plan_box": {"x1": 100, "y1": 200, "x2": 900, "y2": 800}}
BLANK = {"sheet_class": "other", "sheet_number": "", "sheet_title": "", "level": "", "scale": "", "plan_box": {"x1": 0, "y1": 0, "x2": 0, "y2": 0}}
REPLY = {"text": json.dumps(GOOD), "input": 1000, "cached": 0, "output": 100, "finish": "stop"}
UNREAD = {"number": "", "title": "", "cls": "other", "level": "", "scale": "", "viewports": [], "read_ok": False}
REPLY_USD = (1000 * 1.25 + 100 * 4.25) / 1e6


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


def both(chat, sam):
    """Chat calls get `chat`, and SAM calls get `sam`, which is the output text or an exception to raise. Returns each kind's requests."""
    calls = {"chat": [], "sam": []}

    def transport(kind, **request):
        calls[kind].append(request)
        if kind == "chat":
            return chat
        if isinstance(sam, Exception):
            raise sam
        return sam
    meta.set_transport(transport)
    return calls


def always(answer):
    return both(answer, "")["chat"]


def sam_text(*boxes, size=(2000, 1500)):
    """SAM's output for these boxes. x2 and y2 are inclusive in a record, and meta returns them one higher."""
    return "".join(f"<0f>{n}<|box;x1={x1};y1={y1};x2={x2 - 1};y2={y2 - 1};w={size[0]};h={size[1]};c=0.7|><|mask;x=0;y=0;c=0.7;data=2,3,!!!!!&^:o6z|>\n" for n, (x1, y1, x2, y2) in enumerate(boxes))


def sam_image_of(request):
    url = request["input"][0]["content"][1]["image_url"]
    return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)


def viewports(tmp_path, *boxes, sheet=(2000, 1500), size=(2000, 1500)):
    """The viewports of one floor plan sheet on which SAM finds `boxes`."""
    both(REPLY, sam_text(*boxes, size=size))
    return read_stage(make(tmp_path, [sheet]))["sheets"][0]["viewports"]


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
        "plan_box": [100, 100, 900, 400], "viewports": [], "read_ok": True,
    }]}


@pytest.mark.parametrize("bad", BAD.values(), ids=BAD)
def test_empty_or_bad_reply_marks_sheet_unread(tmp_path, bad):
    meta.set_transport(lambda kind, **request: "" if kind == "sam" else {**REPLY, **bad} if sheet_of(request) == 1 else REPLY)
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
        if kind == "sam":
            return ""
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
        if kind == "sam":
            return ""
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


def test_sam_sees_the_2000px_copy_of_a_floor_plan_with_the_concept_prompt(tmp_path):
    calls = both(REPLY, "")
    read_stage(make(tmp_path, [(4000, 3000)]))
    assert len(calls["sam"]) == 1
    assert calls["sam"][0]["input"][0]["content"][0]["text"] == "floor plan"
    assert sam_image_of(calls["sam"][0]).shape == (1500, 2000, 3)


def test_viewports_are_in_sheet_pixels(tmp_path):
    found = viewports(tmp_path, (100, 200, 700, 800), (1200, 200, 1800, 800), sheet=(4000, 3000))
    assert found == [[200, 400, 1400, 1600], [2400, 400, 3600, 1600]]


def test_viewports_on_a_sheet_smaller_than_the_copy_stay_as_they_are(tmp_path):
    assert viewports(tmp_path, (100, 100, 300, 300), sheet=(800, 600), size=(800, 600)) == [[100, 100, 300, 300]]


def test_plan_box_stays_as_spark_gives_it(tmp_path):
    both(REPLY, sam_text((0, 0, 1000, 1000)))
    assert read_stage(make(tmp_path, [(2000, 1500)]))["sheets"][0]["plan_box"] == [200, 300, 1800, 1200]


@pytest.mark.parametrize("cls", [c for c in CLASSES if c != "floor_plan"])
def test_only_a_floor_plan_goes_to_sam(tmp_path, cls):
    calls = both(reply(sheet_class=cls), sam_text((0, 0, 1000, 1000)))
    assert read_stage(make(tmp_path, [(40, 30)]))["sheets"][0]["viewports"] == []
    assert calls["sam"] == []


def test_unread_sheet_does_not_go_to_sam(tmp_path):
    calls = both({"text": "not json"}, sam_text((0, 0, 1000, 1000)))
    assert read_stage(make(tmp_path, [(40, 30)]))["sheets"][0] == UNREAD | {"index": 0, "plan_box": [0, 0, 40, 30]}
    assert calls["sam"] == []


SAM_FAILS = {
    "zero-matches": "",
    "not-sam-output": "no records here",
    "refusal": RuntimeError("SAM response.refusal.done: {}"),
    "stream-cut": RuntimeError("The SAM stream ended before it completed"),
    "bad-request": api_error(openai.BadRequestError, 400),
    "rate-limit": api_error(openai.RateLimitError, 429),
    "server-error": api_error(openai.InternalServerError, 500),
    "gateway-timeout": api_error(openai.APIStatusError, 504),
    "no-connection": openai.APIConnectionError(request=httpx.Request("POST", "http://x")),
}


@pytest.mark.parametrize("sam", SAM_FAILS.values(), ids=SAM_FAILS)
def test_sam_failure_leaves_viewports_empty_and_costs_nothing_else(tmp_path, monkeypatch, sam):
    monkeypatch.setattr(meta.time, "sleep", lambda s: None)
    both(REPLY, sam)
    row = read_stage(make(tmp_path, [(1000, 500)]))["sheets"][0]
    assert row == {"index": 0, "number": "A2.1", "title": "MAIN FLOOR PLAN", "cls": "floor_plan", "level": "GROUND FLOOR", "scale": "1/4\" = 1'-0\"",
                   "plan_box": [100, 100, 900, 400], "viewports": [], "read_ok": True}


def test_sam_failure_on_one_sheet_leaves_the_other_sheets_their_viewports(tmp_path):
    def transport(kind, **request):
        if kind == "chat":
            return REPLY
        if sam_image_of(request)[0, 0, 0] == 1:
            raise RuntimeError("SAM response.failed: {}")
        return sam_text((0, 0, 1000, 1000))
    meta.set_transport(transport)
    rows = read_stage(make(tmp_path, [(2000, 1500)] * 3))["sheets"]
    assert [r["viewports"] for r in rows] == [[[0, 0, 1000, 1000]], [], [[0, 0, 1000, 1000]]]
    assert [r["read_ok"] for r in rows] == [True, True, True]


def test_budget_stop_at_sam_fails_the_stage_like_a_stop_at_spark(tmp_path):
    calls = both(REPLY, sam_text((0, 0, 1000, 1000)))
    with pytest.raises(BudgetReached):
        read_stage(make(tmp_path, [(40, 30)], cap_usd=REPLY_USD / 2))
    assert (len(calls["chat"]), len(calls["sam"])) == (1, 0)


def test_each_floor_plan_adds_one_sam_image_to_the_spend(tmp_path):
    ctx = make(tmp_path, [(40, 30), (40, 30)])
    both(REPLY, "")
    read_stage(ctx)
    assert ctx.meter.spent == pytest.approx(2 * (REPLY_USD + SAM_IMAGE_USD))


def test_a_fresh_run_asks_spark_again_and_keeps_sam_s_cached_answer(tmp_path):
    ctx = make(tmp_path, [(2000, 1500)])
    calls = both(REPLY, sam_text((0, 0, 1000, 1000)))
    first = read_stage(ctx)
    ctx.fresh = True
    assert read_stage(ctx) == first
    assert (len(calls["chat"]), len(calls["sam"])) == (2, 1)


# Boxes SAM gave on real sheets, on the 2000 by 1333 copy. Baxter and USC come from the 0-1000 grid in data/lab/samtricks/item1.json.
REAL = {
    "abbot-a2.0-main-upper-and-a-small-kitchen": ([(1217, 779, 1624, 1227), (1050, 127, 1477, 550), (1616, 153, 1773, 487)], 2),
    "abbot-e2.1-the-closest-storey-stays": ([(894, 775, 1256, 1221), (245, 187, 633, 551), (891, 203, 1255, 538)], 3),
    "baxter-a2.1-the-closest-layouts-go": ([(1212, 136, 1721, 821), (65, 31, 532, 542), (540, 31, 1005, 542)], 1),
    "capitol-a2.1-plan-and-two-layouts": ([(1258, 100, 1643, 993), (544, 30, 994, 329), (71, 30, 458, 328)], 1),
    "capitol-a2.2-plan-and-a-layout": ([(1300, 96, 1651, 993), (542, 63, 914, 503)], 1),
    "capitol-p1.1-four-storeys": ([(1070, 235, 1297, 1070), (681, 236, 906, 1069), (293, 236, 517, 1068), (1462, 239, 1682, 1065)], 4),
    "usc-g003-plan-and-a-key-plan": ([(196, 136, 1313, 947), (1501, 1201, 1806, 1279)], 1),
}


@pytest.mark.parametrize("boxes, plans", REAL.values(), ids=REAL)
def test_real_sam_boxes_count_the_plans_on_the_sheet(tmp_path, boxes, plans):
    assert len(viewports(tmp_path, *boxes, sheet=(2000, 1333), size=(2000, 1333))) == plans


@pytest.mark.parametrize("x2, kept", [(1800, 2), (1799, 1)], ids=["at-the-minimum", "just-under-it"])
def test_a_box_must_reach_the_minimum_share_of_the_largest_box(tmp_path, x2, kept):
    assert len(viewports(tmp_path, (0, 0, 1000, 1000), (1100, 0, x2, 1000))) == kept


def test_a_box_mostly_inside_a_kept_box_counts_once_and_the_larger_stays(tmp_path):
    assert viewports(tmp_path, (0, 200, 900, 1100), (0, 0, 1000, 1000)) == [[0, 0, 1000, 1000]]


def test_boxes_that_share_half_of_the_smaller_or_less_both_count(tmp_path):
    assert viewports(tmp_path, (0, 0, 1000, 1000), (0, 500, 1000, 1500)) == [[0, 0, 1000, 1000], [0, 500, 1000, 1500]]


def test_boxes_apart_on_both_axes_both_count(tmp_path):
    assert viewports(tmp_path, (0, 0, 400, 400), (1600, 1100, 2000, 1500)) == [[0, 0, 400, 400], [1600, 1100, 2000, 1500]]


def test_a_box_past_the_image_sent_does_not_count(tmp_path):
    assert viewports(tmp_path, (0, 0, 1000, 1000), (1100, 0, 2600, 1000), size=(4000, 3000)) == [[0, 0, 1000, 1000]]


def test_a_box_past_the_bottom_of_the_image_sent_does_not_count(tmp_path):
    assert viewports(tmp_path, (0, 0, 1000, 1000), (0, 100, 1000, 1700), size=(4000, 3000)) == [[0, 0, 1000, 1000]]


def test_two_viewports_from_the_read_reach_not_checked(tmp_path):
    both(REPLY, sam_text((100, 100, 900, 1400), (1100, 100, 1900, 1400)))
    lines = evaluate({"read": read_stage(make(tmp_path, [(2000, 1500)]))}, "houston", None, None)["not_checked"]
    assert any(line.startswith("A2.1 holds several floor plans") for line in lines)
