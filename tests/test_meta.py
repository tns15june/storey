import base64
import fcntl
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np
import openai
import pytest

from storey import meta
from storey.meta import BudgetReached, Meter, sam_image, sam_video, spark

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
REPLY = {"text": '{"a": 1}', "input": 1000, "cached": 100, "output": 200, "finish": "stop"}
REPLY_USD = (900 * 1.25 + 100 * 0.15 + 200 * 4.25) / 1e6
BAD = {**REPLY, "text": "not json"}
DAY = 86400

MASK = "data=2,3,!!!!!&^:o6z"   # one_bit, 2 rows by 3 columns: 1 0 1 / 0 1 1
# The video example from Meta's reading-segmentation page, then frame 2 with frame 1 skipped.
VIDEO = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>"
    ",1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n"
    f"<2f>1<|box;x1=10;y1=20;x2=12;y2=21;w=320;h=334;c=0.5|><|mask;x=0;y=0;c=0.5;{MASK}|>\n"
)
IMAGE = f"<0f>7<|box;x1=10;y1=20;x2=12;y2=21;w=100;h=100;c=0.5|><|mask;x=0;y=0;c=0.5;{MASK}|>\n"
VIDEO_ROWS = [
    {"object": 0, "frame": 0, "box": [211, 228, 271, 255], "conf": None},
    {"object": 1, "frame": 0, "box": [155, 228, 203, 255], "conf": None},
    {"object": 1, "frame": 2, "box": [10, 20, 13, 22], "conf": 0.5},
]


class Fake:
    """Stands in for the HTTP call: records each request and gives the next reply, repeating the last one."""

    def __init__(self, *replies):
        self.calls, self.replies = [], list(replies)

    def __call__(self, kind, **request):
        self.calls.append((kind, request))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def of(self, kind):
        return [r for k, r in self.calls if k == kind]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("a test reached the real client"))
    yield
    meta.set_transport(None)


@pytest.fixture
def m(tmp_path):
    return Meter(ledger=tmp_path / "ledger.jsonl")


def fake(*replies):
    f = Fake(*replies)
    meta.set_transport(f)
    return f


def rows(meter):
    return [json.loads(line) for line in meter.ledger.read_text().splitlines()]


def api_status(code):
    return openai.APIStatusError("no", response=httpx.Response(code, request=httpx.Request("POST", "http://x")), body=None)


# Spark, cache and spend


def test_cache_hit_costs_nothing(m):
    f = fake(REPLY)
    assert spark("q", meter=m, schema=SCHEMA) == {"a": 1}
    assert m.spent == pytest.approx(REPLY_USD)
    assert spark("q", meter=m, schema=SCHEMA) == {"a": 1}
    assert len(f.calls) == 1
    assert m.spent == pytest.approx(REPLY_USD)
    assert len(rows(m)) == 1


def test_token_cap_always_sent(m):
    f = fake(REPLY)
    spark("q", meter=m)
    assert f.calls[0][1]["max_completion_tokens"] == 24000


def test_budget_stops_before_call(tmp_path):
    f = fake(REPLY)
    m = Meter(cap_usd=1, ledger=tmp_path / "ledger.jsonl")
    m.add(1.0)
    with pytest.raises(BudgetReached):
        spark("q", meter=m)
    assert f.calls == []


def test_cached_reply_is_free_even_at_the_cap(tmp_path):
    fake(REPLY)
    m = Meter(cap_usd=1, ledger=tmp_path / "ledger.jsonl")
    spark("q", meter=m)
    m.add(1.0)
    assert spark("q", meter=m) == REPLY["text"]


def test_daily_cap_counts_other_reviews(tmp_path):
    f = fake(REPLY)
    ledger = tmp_path / "ledger.jsonl"
    other = Meter(cap_usd=100, daily_cap_usd=100, ledger=ledger)
    for _ in range(5):
        other.add(5.0, model="muse-spark-1.3")
    m = Meter(ledger=ledger)
    assert m.spent == 0
    with pytest.raises(BudgetReached):
        m.check()
    with pytest.raises(BudgetReached):
        spark("q", meter=m)
    assert f.calls == []


@pytest.mark.parametrize("t, stopped", [(-1, False), (0, True), (DAY - 1, True)])
def test_daily_cap_reads_the_current_utc_day(tmp_path, monkeypatch, t, stopped):
    midnight = 20000 * DAY
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"t": midnight + t, "usd": 25.0}) + "\n")
    monkeypatch.setattr(meta.time, "time", lambda: midnight + 60)
    if stopped:
        with pytest.raises(BudgetReached):
            Meter(ledger=ledger).check()
    else:
        Meter(ledger=ledger).check()


def test_torn_ledger_line_is_skipped(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    now = 20000 * DAY
    ledger.write_text('{"t": 17\n' + json.dumps({"t": now, "usd": 25.0}) + "\n")
    monkeypatch.setattr(meta.time, "time", lambda: now)
    with pytest.raises(BudgetReached):
        Meter(ledger=ledger).check()


def test_missing_ledger_means_nothing_spent(m):
    m.check()
    assert m.spent == 0


def test_budget_message_is_plain_and_names_the_cap(tmp_path):
    m = Meter(cap_usd=2, ledger=tmp_path / "ledger.jsonl")
    m.add(2.0)
    with pytest.raises(BudgetReached, match="2.00 USD"):
        m.check()


def test_malformed_json_is_retried_once_then_raises_valueerror(m):
    f = fake(BAD)
    with pytest.raises(ValueError):
        spark("q", meter=m, schema=SCHEMA)
    assert len(f.calls) == 2
    assert m.spent == pytest.approx(2 * REPLY_USD)
    with pytest.raises(ValueError):   # a bad reply is never cached
        spark("q", meter=m, schema=SCHEMA)
    assert len(f.calls) == 4


def test_retry_that_gets_a_good_reply_is_returned_and_cached(m):
    f = fake(BAD, REPLY)
    assert spark("q", meter=m, schema=SCHEMA) == {"a": 1}
    assert spark("q", meter=m, schema=SCHEMA) == {"a": 1}
    assert len(f.calls) == 2


def test_retry_is_refused_once_the_budget_is_gone(tmp_path):
    f = fake({**BAD, "input": 1_000_000, "cached": 0})
    m = Meter(cap_usd=1, ledger=tmp_path / "ledger.jsonl")
    with pytest.raises(BudgetReached):
        spark("q", meter=m, schema=SCHEMA)
    assert len(f.calls) == 1


@pytest.mark.parametrize("text", [None, "", "  \n"])
@pytest.mark.parametrize("schema", [SCHEMA, None])
def test_empty_reply_raises_valueerror(m, text, schema):
    f = fake({**REPLY, "text": text})
    with pytest.raises(ValueError):
        spark("q", meter=m, schema=schema)
    assert len(f.calls) == 2


@pytest.mark.parametrize("text", ["[1, 2]", "null", "3"])
def test_schema_reply_that_is_not_an_object_raises_valueerror(m, text):
    fake({**REPLY, "text": text})
    with pytest.raises(ValueError):
        spark("q", meter=m, schema=SCHEMA)


@pytest.mark.parametrize("finish", ["length", "content_filter"])
@pytest.mark.parametrize("schema, text", [(SCHEMA, '{"a": 1'), (None, "half a sent")])
def test_reply_cut_off_at_the_token_cap_is_not_retried(m, schema, text, finish):
    f = fake({**REPLY, "text": text, "finish": finish})
    with pytest.raises(ValueError):
        spark("q", meter=m, schema=schema)
    assert len(f.calls) == 1
    assert m.spent == pytest.approx(REPLY_USD)
    with pytest.raises(ValueError):
        spark("q", meter=m, schema=schema)
    assert len(f.calls) == 2


def test_fresh_skips_the_cache_and_replaces_it(m):
    f = fake({**REPLY, "text": "old"}, {**REPLY, "text": "new"})
    assert spark("q", meter=m) == "old"
    assert spark("q", meter=m, fresh=True) == "new"
    assert spark("q", meter=m) == "new"
    assert len(f.calls) == 2


@pytest.mark.parametrize("change", [
    {"prompt": "other"},
    {"model": "muse-wonder-1.0"},
    {"effort": "medium"},
    {"system": "be brief"},
    {"schema": SCHEMA},
    {"images": [np.ones((2, 2, 3), np.uint8)]},
])
def test_each_input_changes_the_cache_key(m, change):
    f = fake(REPLY)
    base = {"prompt": "q", "images": [np.zeros((2, 2, 3), np.uint8)]}
    spark(**base, meter=m)
    spark(**{**base, **change}, meter=m)
    assert len(f.calls) == 2


def test_token_cap_is_not_part_of_the_cache_key(m):
    f = fake(REPLY)
    spark("q", meter=m)
    spark("q", meter=m, max_tokens=8000)
    assert len(f.calls) == 1


def test_request_carries_system_effort_and_schema(m):
    f = fake(REPLY)
    spark("q", meter=m, schema=SCHEMA, system="be brief", effort="medium", model="muse-wonder-1.0", max_tokens=8000)
    r = f.calls[0][1]
    assert r["model"] == "muse-wonder-1.0"
    assert r["reasoning_effort"] == "medium"
    assert r["max_completion_tokens"] == 8000
    assert r["messages"][0] == {"role": "system", "content": "be brief"}
    assert r["messages"][1]["content"] == [{"type": "text", "text": "q"}]
    assert r["response_format"] == {"type": "json_schema", "json_schema": {"name": "out", "schema": SCHEMA, "strict": True}}


@pytest.mark.parametrize("model, usd", [
    ("muse-spark-1.3", (900 * 1.25 + 100 * 0.15 + 200 * 4.25) / 1e6),
    ("muse-wonder-1.0", (900 * 4.0 + 100 * 0.20 + 200 * 20.0) / 1e6),
])
def test_cost_uses_the_model_prices(m, model, usd):
    fake(REPLY)
    spark("q", meter=m, model=model)
    assert m.spent == pytest.approx(usd)
    row = rows(m)[0]
    assert (row["model"], row["input"], row["output"]) == (model, 1000, 200)


def test_contributor_model_never_reaches_the_api(m):
    f = fake(REPLY)
    with pytest.raises(KeyError):
        spark("q", meter=m, model="muse-spark-1.3-contributor")
    assert f.calls == []


def test_array_goes_as_png_with_exact_pixels(m):
    f = fake(REPLY)
    im = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    spark("q", [im], meter=m)
    url = f.calls[0][1]["messages"][0]["content"][1]["image_url"]["url"]
    head, data = url.split(",", 1)
    assert head == "data:image/png;base64"
    assert np.array_equal(cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR), im)


@pytest.mark.parametrize("name, mime", [("a.png", "image/png"), ("a.JPG", "image/jpeg"), ("a.jpeg", "image/jpeg"), ("a.webp", "image/webp")])
def test_file_bytes_pass_through(m, tmp_path, name, mime):
    f = fake(REPLY)
    path = tmp_path / name
    path.write_bytes(b"file bytes")
    spark("q", [path], meter=m)
    url = f.calls[0][1]["messages"][0]["content"][1]["image_url"]["url"]
    assert url == f"data:{mime};base64," + base64.b64encode(b"file bytes").decode()


def test_retries_a_rate_limit_then_succeeds(m, monkeypatch):
    monkeypatch.setattr(meta.time, "sleep", lambda s: None)
    f = fake(api_status(429), api_status(503), REPLY)
    assert spark("q", meter=m) == REPLY["text"]
    assert len(f.calls) == 3
    assert m.spent == pytest.approx(REPLY_USD)


def test_retries_a_dropped_connection(m, monkeypatch):
    monkeypatch.setattr(meta.time, "sleep", lambda s: None)
    f = fake(openai.APIConnectionError(request=httpx.Request("POST", "http://x")), REPLY)
    assert spark("q", meter=m) == REPLY["text"]
    assert len(f.calls) == 2


@pytest.mark.parametrize("code", [400, 504])
def test_status_that_would_fail_again_is_not_retried(m, monkeypatch, code):
    monkeypatch.setattr(meta.time, "sleep", lambda s: pytest.fail("slept"))
    f = fake(api_status(code))
    with pytest.raises(openai.APIStatusError):
        spark("q", meter=m)
    assert len(f.calls) == 1


def test_gives_up_after_six_tries_waiting_longer_each_time(m, monkeypatch):
    sleeps = []
    monkeypatch.setattr(meta.time, "sleep", sleeps.append)
    monkeypatch.setattr(meta.random, "random", lambda: 0.25)   # real jitter can reverse two neighbouring waits
    f = fake(api_status(500))
    with pytest.raises(openai.APIStatusError):
        spark("q", meter=m)
    assert len(f.calls) == 6
    assert sleeps == [0.75, 1.25, 2.25, 4.25, 8.25]
    assert m.spent == 0


# Threads


def test_parallel_calls_lose_no_spend(tmp_path):
    fake(REPLY)
    m = Meter(cap_usd=1000, daily_cap_usd=1000, ledger=tmp_path / "ledger.jsonl")
    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(lambda i: spark(f"q{i}", meter=m, schema=SCHEMA), range(400)))
    assert out == [{"a": 1}] * 400
    assert m.spent == pytest.approx(400 * REPLY_USD)
    assert sum(r["usd"] for r in rows(m)) == pytest.approx(400 * REPLY_USD, abs=1e-3)
    assert len(rows(m)) == 400


@pytest.mark.parametrize("call", [lambda m: m.add(1.0), lambda m: m.check()], ids=["write", "read"])
def test_ledger_waits_for_the_file_lock(m, call):
    m.ledger.touch()
    with open(m.ledger) as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        worker = threading.Thread(target=call, args=(m,), daemon=True)
        worker.start()
        worker.join(0.2)
        assert worker.is_alive()
    worker.join(5)
    assert not worker.is_alive()


def test_threads_racing_on_one_key_leave_one_good_entry(m, tmp_path):
    fake(REPLY)
    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(lambda i: spark("same", meter=m, schema=SCHEMA), range(32)))
    assert out == [{"a": 1}] * 32
    entries = list((tmp_path / "cache").iterdir())
    assert len(entries) == 1
    assert entries[0].read_text() == REPLY["text"]


def test_half_written_cache_entry_is_never_visible(monkeypatch):
    write_text, seen = Path.write_text, []

    def spy(path, text, **kwargs):
        write_text(path, text[:1], **kwargs)
        seen.append(meta._cached("key"))
        write_text(path, text, **kwargs)
    monkeypatch.setattr(Path, "write_text", spy)
    meta._store("key", "whole reply")
    assert seen == [None]
    assert meta._cached("key") == "whole reply"


# SAM


def test_sam_video_rows(m, tmp_path):
    path = tmp_path / "stack.mp4"
    path.write_bytes(b"video bytes")
    f = fake("file-1", VIDEO)
    assert sam_video("<0f>1<|box;x1=1;y1=2;x2=3;y2=4;w=9;h=9|>", path, 6, meter=m) == VIDEO_ROWS
    assert f.calls[0] == ("upload", {"purpose": "user_data", "file": ("stack.mp4", b"video bytes")})
    kind, r = f.calls[1]
    assert (kind, r["model"]) == ("sam", "sam-3.1")
    assert r["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit", "propagation_direction": "both"}
    assert r["input"][0]["content"] == [
        {"type": "input_text", "text": "<0f>1<|box;x1=1;y1=2;x2=3;y2=4;w=9;h=9|>"},
        {"type": "input_video", "file_id": "file-1"},
    ]


def test_sam_video_object_ids_are_ints(m, tmp_path):
    (tmp_path / "s.mp4").write_bytes(b"v")
    fake("file-1", VIDEO)
    assert all(type(row["object"]) is int for row in sam_video("p", tmp_path / "s.mp4", 6, meter=m))


@pytest.mark.parametrize("frames, usd", [(6, 50 * 0.0002), (50, 50 * 0.0002), (120, 120 * 0.0002)])
def test_sam_video_cost_has_a_50_frame_floor(m, tmp_path, frames, usd):
    (tmp_path / "s.mp4").write_bytes(b"v")
    fake("file-1", VIDEO)
    sam_video("p", tmp_path / "s.mp4", frames, meter=m)
    assert m.spent == pytest.approx(usd)
    assert (rows(m)[0]["model"], rows(m)[0]["kind"], rows(m)[0]["frames"]) == ("sam-3.1", "video", frames)


def test_sam_video_cache_hit_is_free_and_skips_the_upload(m, tmp_path):
    (tmp_path / "s.mp4").write_bytes(b"v")
    f = fake("file-1", VIDEO)
    sam_video("p", tmp_path / "s.mp4", 6, meter=m)
    spent = m.spent
    assert sam_video("p", tmp_path / "s.mp4", 6, meter=m) == VIDEO_ROWS
    assert len(f.calls) == 2
    assert m.spent == spent


def test_sam_video_direction_is_in_the_cache_key(m, tmp_path):
    (tmp_path / "s.mp4").write_bytes(b"v")
    f = fake("file-1", VIDEO, "file-2", VIDEO)
    sam_video("p", tmp_path / "s.mp4", 6, meter=m)
    sam_video("p", tmp_path / "s.mp4", 6, meter=m, direction="forward")
    assert [r["metadata"]["propagation_direction"] for r in f.of("sam")] == ["both", "forward"]


@pytest.mark.parametrize("prompt, blob", [("other", b"v"), ("p", b"other bytes")], ids=["prompt", "file-bytes"])
def test_sam_video_prompt_and_file_bytes_are_in_the_cache_key(m, tmp_path, prompt, blob):
    path = tmp_path / "s.mp4"
    path.write_bytes(b"v")
    f = fake("file-1", VIDEO, "file-2", VIDEO)
    sam_video("p", path, 6, meter=m)
    path.write_bytes(blob)
    sam_video(prompt, path, 6, meter=m)
    assert len(f.of("sam")) == 2


def test_sam_video_budget_stops_before_the_upload(tmp_path):
    (tmp_path / "s.mp4").write_bytes(b"v")
    f = fake("file-1", VIDEO)
    m = Meter(cap_usd=1, ledger=tmp_path / "ledger.jsonl")
    m.add(1.0)
    with pytest.raises(BudgetReached):
        sam_video("p", tmp_path / "s.mp4", 6, meter=m)
    assert f.calls == []


def test_sam_image_rows(m):
    f = fake(IMAGE)
    out = sam_image("door", np.zeros((4, 4, 3), np.uint8), meter=m)
    assert [{k: row[k] for k in ("object", "box", "conf")} for row in out] == [{"object": 7, "box": [10, 20, 13, 22], "conf": 0.5}]
    assert type(out[0]["object"]) is int
    assert out[0]["mask"].dtype == bool
    assert out[0]["mask"].tolist() == [[True, False, True], [False, True, True]]
    r = f.calls[0][1]
    assert r["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit"}
    assert r["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert m.spent == pytest.approx(0.0025)
    assert sam_image("door", np.zeros((4, 4, 3), np.uint8), meter=m)[0]["mask"].tolist() == out[0]["mask"].tolist()
    assert len(f.calls) == 1
    assert m.spent == pytest.approx(0.0025)


def test_sam_image_mask_is_scaled_to_its_box(m):
    fake(f"<0f>1<|box;x1=0;y1=0;x2=5;y2=3;w=100;h=100|><|mask;x=0;y=0;{MASK}|>\n")
    mask = sam_image("door", np.zeros((4, 4, 3), np.uint8), meter=m)[0]["mask"]
    assert mask.shape == (4, 6)
    assert mask[0].tolist() == [True, True, False, False, True, True]


@pytest.mark.parametrize("change", [{"prompt": "window"}, {"image": np.ones((4, 4, 3), np.uint8)}], ids=["prompt", "image"])
def test_sam_image_prompt_and_image_are_in_the_cache_key(m, change):
    f = fake(IMAGE)
    base = {"prompt": "door", "image": np.zeros((4, 4, 3), np.uint8)}
    sam_image(**base, meter=m)
    sam_image(**{**base, **change}, meter=m)
    assert len(f.calls) == 2


def test_sam_budget_stops_before_call(tmp_path):
    f = fake(IMAGE)
    m = Meter(cap_usd=1, ledger=tmp_path / "ledger.jsonl")
    m.add(1.0)
    with pytest.raises(BudgetReached):
        sam_image("door", np.zeros((4, 4, 3), np.uint8), meter=m)
    assert f.calls == []


def test_zero_matches_is_a_valid_cached_reply(m):
    f = fake("")
    assert sam_image("door", np.zeros((4, 4, 3), np.uint8), meter=m) == []
    assert sam_image("door", np.zeros((4, 4, 3), np.uint8), meter=m) == []
    assert len(f.calls) == 1


@pytest.mark.parametrize("text", [
    "<3f>\n",
    "I cannot segment this drawing.\n",
    "<0f>1<|box;x1=1;y1=2;x2=3;y2=4;w=9;h=9|><|mask;x=0;y=0;data=2,3,!bad|>\n",
    IMAGE + "<1f>1<|box;x1=1;y1=2;x2=3;y2=4;w=9;h=9|>\n",
])
def test_malformed_sam_reply_raises_valueerror_and_is_not_cached(m, text):
    f = fake(text)
    im = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ValueError):
        sam_image("door", im, meter=m)
    assert m.spent == pytest.approx(0.0025)
    with pytest.raises(ValueError):
        sam_image("door", im, meter=m)
    assert len(f.calls) == 2


# The real OpenAI client over a mock HTTP layer: the SDK builds each request and parses each reply, and nothing leaves the process.


def serve(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    client = openai.OpenAI(base_url=meta.BASE, api_key="test", max_retries=0, http_client=httpx.Client(transport=transport))
    monkeypatch.setattr(meta, "_client", lambda: client)


def serve_json(monkeypatch, body):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=body)
    serve(monkeypatch, handler)
    return seen


def chat_body(content="hi", finish="stop", **usage):
    return {
        "id": "c", "object": "chat.completion", "created": 1, "model": "muse-spark-1.3",
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, **usage},
    }


def delta(text):
    return {"type": "response.output_text.delta", "item_id": "i", "output_index": 0, "content_index": 0, "delta": text}


def part_done(**part):
    return {"type": "response.content_part.done", "item_id": "i", "output_index": 0, "content_index": 0, "part": part}


def sse(*events):
    text = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)


COMPLETED = {"type": "response.completed", "response": {"id": "r", "status": "completed"}}
FILE = {"id": "file-9", "object": "file", "bytes": 11, "created_at": 1, "filename": "s.mp4", "purpose": "user_data"}


def test_real_chat_call_reads_text_and_usage(m, monkeypatch):
    seen = serve_json(monkeypatch, chat_body(prompt_tokens_details={"cached_tokens": 3}))
    assert spark("q", meter=m) == "hi"
    sent = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert (sent["max_completion_tokens"], sent["reasoning_effort"]) == (24000, "low")
    assert m.spent == pytest.approx((7 * 1.25 + 3 * 0.15 + 4 * 4.25) / 1e6)


def test_real_chat_call_without_cache_details(m, monkeypatch):
    serve_json(monkeypatch, chat_body())
    spark("q", meter=m)
    assert m.spent == pytest.approx((10 * 1.25 + 4 * 4.25) / 1e6)


@pytest.mark.parametrize("content, finish", [('{"a"', "length"), (None, "content_filter")], ids=["length", "filtered"])
def test_real_chat_call_cut_off_is_not_retried(m, monkeypatch, content, finish):
    seen = serve_json(monkeypatch, chat_body(content=content, finish=finish))
    with pytest.raises(ValueError):
        spark("q", meter=m, schema=SCHEMA)
    assert len(seen) == 1


@pytest.mark.parametrize("body", [{**chat_body(), "choices": []}, {"id": "c"}], ids=["no-choices", "bare"])
def test_real_chat_call_with_no_content_is_an_empty_reply(m, monkeypatch, body):
    seen = serve_json(monkeypatch, body)
    with pytest.raises(ValueError):
        spark("q", meter=m, schema=SCHEMA)
    assert len(seen) == 2


def test_real_rate_limit_is_retried(m, monkeypatch):
    monkeypatch.setattr(meta.time, "sleep", lambda s: None)
    replies = [httpx.Response(429, json={"error": {"message": "slow down"}}), httpx.Response(200, json=chat_body())]
    serve(monkeypatch, lambda request: replies.pop(0))
    assert spark("q", meter=m) == "hi"


def test_real_sam_call_joins_the_stream(m, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return sse(delta(IMAGE[:40]), delta(IMAGE[40:]), COMPLETED)
    serve(monkeypatch, handler)
    assert sam_image("door", np.zeros((4, 4, 3), np.uint8), meter=m)[0]["object"] == 7
    sent = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/responses"
    assert sent["stream"] is True
    assert sent["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit"}


def test_real_sam_call_with_no_output_is_zero_matches(m, monkeypatch):
    serve(monkeypatch, lambda request: sse(COMPLETED))
    assert sam_image("door", np.zeros((4, 4, 3), np.uint8), meter=m) == []


@pytest.mark.parametrize("deltas", [[], [delta(IMAGE[:40]), delta(IMAGE[40:])]], ids=["final-part-only", "deltas-then-final-part"])
def test_real_sam_final_part_gives_the_text_once(m, monkeypatch, deltas):
    serve(monkeypatch, lambda request: sse(*deltas, part_done(type="output_text", text=IMAGE), COMPLETED))
    assert [row["object"] for row in sam_image("door", np.zeros((4, 4, 3), np.uint8), meter=m)] == [7]


def test_real_sam_refusal_with_no_text_is_not_zero_matches(m, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return sse(part_done(type="refusal", refusal="no"), COMPLETED)
    serve(monkeypatch, handler)
    im = np.zeros((4, 4, 3), np.uint8)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            sam_image("door", im, meter=m)
    assert len(seen) == 2
    assert m.spent == 0


def test_real_sam_shutdown_mid_stream_replays_the_whole_request(m, monkeypatch):
    monkeypatch.setattr(meta.time, "sleep", lambda s: None)
    streams = [sse(delta(IMAGE[:40]), {"type": "error", "code": "server_shutting_down", "message": "draining"}), sse(delta(IMAGE), COMPLETED)]
    serve(monkeypatch, lambda request: streams.pop(0))
    assert [row["object"] for row in sam_image("door", np.zeros((4, 4, 3), np.uint8), meter=m)] == [7]
    assert streams == []
    assert m.spent == pytest.approx(0.0025)


@pytest.mark.parametrize("end", [
    [{"type": "response.failed", "response": {"id": "r", "status": "failed"}}],
    [{"type": "response.incomplete", "response": {"id": "r", "status": "incomplete"}}],
    [{"type": "response.refusal.done", "refusal": "no"}, COMPLETED],
    [{"type": "response.refusal.delta", "delta": "no"}, COMPLETED],
    [part_done(type="refusal", refusal="no"), COMPLETED],
    [{"type": "error", "message": "boom"}],
    [{"type": "error", "error": {"message": "boom"}}],
    [],
], ids=["failed", "incomplete", "refusal-done", "refusal-delta", "refusal-part", "error", "error-object", "no-end"])
def test_real_sam_stream_that_does_not_complete_raises_and_is_not_cached(m, monkeypatch, end):
    seen = []

    def handler(request):
        seen.append(request)
        return sse(delta(IMAGE), *end)
    serve(monkeypatch, handler)
    im = np.zeros((4, 4, 3), np.uint8)
    for _ in range(2):
        with pytest.raises((RuntimeError, openai.APIError)):
            sam_image("door", im, meter=m)
    assert len(seen) == 2
    assert m.spent == 0


def test_real_video_call_uploads_bytes_and_sends_the_file_id(m, monkeypatch, tmp_path):
    (tmp_path / "s.mp4").write_bytes(b"video bytes")
    seen = {}

    def handler(request):
        seen[request.url.path] = request
        return httpx.Response(200, json=FILE) if request.url.path == "/v1/files" else sse(delta(VIDEO), COMPLETED)
    serve(monkeypatch, handler)
    assert sam_video("p", tmp_path / "s.mp4", 6, meter=m) == VIDEO_ROWS
    assert b"video bytes" in seen["/v1/files"].content
    assert b"user_data" in seen["/v1/files"].content
    sent = json.loads(seen["/v1/responses"].content)
    assert sent["input"][0]["content"][1] == {"type": "input_video", "file_id": "file-9"}
    assert sent["metadata"]["propagation_direction"] == "both"


def test_real_retried_upload_sends_the_bytes_again(m, monkeypatch, tmp_path):
    monkeypatch.setattr(meta.time, "sleep", lambda s: None)
    (tmp_path / "s.mp4").write_bytes(b"video bytes")
    uploads = []

    def handler(request):
        if request.url.path != "/v1/files":
            return sse(COMPLETED)
        uploads.append(request.content)
        return httpx.Response(503, json={"error": {"message": "busy"}}) if len(uploads) == 1 else httpx.Response(200, json=FILE)
    serve(monkeypatch, handler)
    assert sam_video("p", tmp_path / "s.mp4", 6, meter=m) == []
    assert len(uploads) == 2
    assert all(b"video bytes" in body for body in uploads)


# Key


def test_key_comes_from_the_environment_first_then_dotenv(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "ROOT", tmp_path)
    (tmp_path / ".env").write_text('OTHER=1\nMODEL_API_KEY="from-file"\n')
    monkeypatch.setenv("MODEL_API_KEY", "from-env")
    assert meta._api_key() == "from-env"
    monkeypatch.delenv("MODEL_API_KEY")
    assert meta._api_key() == "from-file"


def test_missing_key_raises_a_plain_error(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "ROOT", tmp_path)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        meta._api_key()
