import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from storey import meta, pipeline, server
from storey.meta import Meter, spark
from storey.pipeline import Context, read_json, run, write_json

SHEETS = [{"index": 0, "image": "sheet_000.png"}]
DEAR = {"text": "ok", "input": 7_000_000, "output": 0}   # 8.75 USD, past the 8 USD cap


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("a test reached the real client"))
    yield
    meta.set_transport(None)


@pytest.fixture
def review(tmp_path):
    path = tmp_path / "reviews" / "r1"
    path.mkdir(parents=True)
    (path / "sheets.json").write_text(json.dumps(SHEETS))
    return path


def stage(name, fn=lambda ctx: {}):
    return (name, name.title(), fn)


def status(path):
    return json.loads((path / "status.json").read_text())


def stored(path, name):
    return json.loads((path / f"stage_{name}.json").read_text())


def states(path):
    return [row["state"] for row in status(path)["stages"]]


def boom(ctx):
    raise KeyError("plan_box")


def test_stages_run_in_order(review):
    seen = []
    run(review, stages=[stage(n, lambda ctx, n=n: seen.append(n) or {"name": n}) for n in ("read", "floors", "letter")])
    assert seen == ["read", "floors", "letter"]
    assert stored(review, "floors") == {"name": "floors"}


def test_a_stage_sees_the_sheets_and_stored_results(review):
    seen = {}

    def second(ctx):
        seen.update(sheets=ctx.sheets, first=ctx.stage("first"), later=ctx.stage("third"), fresh=ctx.fresh)
        return {}
    run(review, stages=[stage("first", lambda ctx: {"a": 1}), stage("second", second), stage("third")])
    assert seen == {"sheets": SHEETS, "first": {"a": 1}, "later": {}, "fresh": False}


def test_finished_stage_is_skipped_on_rerun(review):
    calls = []
    stages = [stage("empty", lambda ctx: calls.append("empty") or {}), stage("full", lambda ctx: calls.append("full") or {"x": 1})]
    run(review, stages=stages)
    run(review, stages=stages)
    assert calls == ["empty", "full"]
    assert status(review)["state"] == "done"
    assert states(review) == ["done", "done"]


def test_rerun_after_a_stop_runs_only_the_missing_stage(review):
    calls = []
    stages = [stage(n, lambda ctx, n=n: calls.append(n) or {}) for n in ("a", "b", "c")]
    run(review, stages=stages)
    (review / "stage_b.json").unlink()
    run(review, stages=stages)
    assert calls == ["a", "b", "c", "b"]


def test_fresh_reruns_finished_stages_and_reaches_them(review):
    flags = []
    stages = [stage("a", lambda ctx: flags.append(ctx.fresh) or {})]
    run(review, stages=stages)
    run(review, stages=stages, fresh=True)
    assert flags == [False, True]


def test_empty_stage_list_means_no_stages(review, monkeypatch):
    monkeypatch.setattr(pipeline, "STAGES", [stage("default")])
    run(review, stages=[])
    assert not (review / "stage_default.json").exists()
    run(review)
    assert stored(review, "default") == {}


def test_failed_stage_does_not_stop_later_stages(review):
    later = []
    run(review, stages=[stage("a"), stage("b", boom), stage("c", lambda ctx: later.append(ctx.stage("b")) or {})])
    assert later == [stored(review, "b")]
    assert "failed" in stored(review, "b")
    assert status(review)["state"] == "done"
    assert states(review) == ["done", "failed", "done"]


def test_failure_message_is_plain_and_the_exception_text_goes_nowhere(review, caplog):
    text = 'SAM response.failed: {"detail": "sk-not-a-real-key"}'

    def dumps_a_reply(ctx):
        raise RuntimeError(text)
    with caplog.at_level(logging.ERROR, logger="storey.pipeline"):
        run(review, stages=[("read", "Read the sheets", dumps_a_reply)])
    assert stored(review, "read") == {"failed": "Read the sheets did not finish."}
    assert "RuntimeError" in caplog.text and "dumps_a_reply" in caplog.text
    assert "sk-not-a-real-key" not in caplog.text


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf"), float("-inf")], ids=["object", "nan", "inf", "-inf"])
def test_result_that_is_not_json_fails_only_that_stage(review, value):
    run(review, stages=[stage("a", lambda ctx: {"clear_in": value}), stage("b", lambda ctx: {"ok": True})])
    assert stored(review, "a") == {"failed": "A did not finish."}
    assert stored(review, "b") == {"ok": True}
    assert states(review) == ["failed", "done"]


def test_stored_dict_keys_come_back_as_strings(review):
    seen = []
    stages = [stage("floors", lambda ctx: {"order": [3, 5], "inliers": {3: 40, 5: 31}, "marks": {"12": "bifold"}}),
              stage("verticals", lambda ctx: seen.append(ctx.stage("floors")) or {})]
    run(review, stages=stages)
    assert seen == [{"order": [3, 5], "inliers": {"3": 40, "5": 31}, "marks": {"12": "bifold"}}]


def test_failed_stage_is_retried_on_rerun_and_later_stages_follow(review):
    outcomes = [ValueError("no"), {"good": 1}]
    calls = []

    def flaky(ctx):
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out
    stages = [stage("a", lambda ctx: calls.append("a") or {}), stage("b", flaky), stage("c", lambda ctx: calls.append("c") or {"b": ctx.stage("b")})]
    run(review, stages=stages)
    assert stored(review, "c") == {"b": stored(review, "b")} and "failed" in stored(review, "b")
    run(review, stages=stages)
    assert calls == ["a", "c", "c"]
    assert stored(review, "c") == {"b": {"good": 1}}
    assert states(review) == ["done", "done", "done"]


def spender(ctx):
    spark("first", meter=ctx.meter)
    spark("second", meter=ctx.meter)
    return {"never": "stored"}


def test_budget_stop_is_recorded_and_review_finishes(review):
    meta.set_transport(lambda kind, **request: DEAR)
    run(review, stages=[stage("read", lambda ctx: {"kept": True}), stage("verticals", spender), stage("vector", lambda ctx: {"ran": True})])
    assert "8.00 USD" in stored(review, "verticals")["failed"]
    assert stored(review, "read") == {"kept": True}
    assert stored(review, "vector") == {"ran": True}
    assert status(review)["state"] == "done"
    assert states(review) == ["done", "failed", "done"]
    assert status(review)["usd"] == pytest.approx(8.75)


def test_the_cap_holds_across_reruns_and_no_call_is_paid_twice(review):
    calls = []

    def transport(kind, **request):
        calls.append(request)
        return DEAR
    meta.set_transport(transport)
    stages = [stage("verticals", spender)]
    run(review, stages=stages)
    run(review, stages=stages)
    assert len(calls) == 1   # the first call comes from the cache and the cap stops the second
    assert "8.00 USD" in stored(review, "verticals")["failed"]
    assert status(review)["usd"] == pytest.approx(8.75)


def test_daily_cap_counts_every_review_folder(review):
    (review.parent / "ledger.jsonl").write_text(json.dumps({"t": time.time(), "usd": 25.0}) + "\n")
    run(review, stages=[stage("read", lambda ctx: ctx.meter.check() or {})])
    assert "daily" in stored(review, "read")["failed"]


def test_status_has_each_stage(review):
    seen = []

    def reading(ctx):
        ctx.progress("Reading sheet 2 of 5", 2, 5)
        seen.append(status(review))
        return {}
    run(review, stages=[("read", "Read the sheets", reading), ("floors", "Line up the floors", lambda ctx: {})])
    assert seen[0] == {
        "state": "running",
        "step": "Reading sheet 2 of 5",
        "stages": [
            {"name": "read", "label": "Read the sheets", "state": "running", "done": 2, "total": 5},
            {"name": "floors", "label": "Line up the floors", "state": "queued", "done": 0, "total": 0},
        ],
        "usd": 0,
    }
    assert status(review) == {
        "state": "done",
        "step": "Done",
        "stages": [
            {"name": "read", "label": "Read the sheets", "state": "done", "done": 2, "total": 5},
            {"name": "floors", "label": "Line up the floors", "state": "done", "done": 0, "total": 0},
        ],
        "usd": 0,
    }


def test_progress_is_safe_from_worker_threads(review):
    def parallel(ctx):
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda i: ctx.progress("Reading", i + 1, 400), range(400)))
        return {"ok": True}
    run(review, stages=[stage("read", parallel)])
    assert stored(review, "read") == {"ok": True}
    assert not list(review.glob("*.tmp"))


def test_context_works_outside_a_run(tmp_path):
    ctx = Context(tmp_path, Meter(ledger=tmp_path / "ledger.jsonl"), [], False)
    assert ctx.stage("read") == {}
    ctx.progress("Reading", 1, 2)
    assert status(tmp_path)["step"] == "Reading"


def test_half_written_json_is_never_visible(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    write_json(target, {"old": True})
    write_text, seen = Path.write_text, []

    def spy(path, text, **kwargs):
        write_text(path, text[:1], **kwargs)
        seen.append(read_json(target))
        write_text(path, text, **kwargs)
    monkeypatch.setattr(Path, "write_text", spy)
    write_json(target, {"new": True})
    assert seen == [{"old": True}]
    assert read_json(target) == {"new": True}


def test_missing_json_file_reads_as_empty(tmp_path):
    assert read_json(tmp_path / "none.json") == {}


# The server around the pipeline


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA", tmp_path / "reviews")
    monkeypatch.setenv("STOREY_ACCESS_KEY", "test-key")
    return TestClient(server.app, base_url="http://127.0.0.1", headers={"x-storey-key": "test-key"})


def stub_ingest(monkeypatch, sheets=json.dumps(SHEETS)):
    code = ("import sys, json, pathlib; d = pathlib.Path(sys.argv[1]); "
            f"(d / 'sheets.json').write_text({sheets!r}); (d / 'ingest_result.json').write_text(json.dumps({{'sheets': 1}}))")
    monkeypatch.setattr(server, "INGEST_COMMAND", [sys.executable, "-c", code])


def upload(client):
    return client.post("/api/reviews", files=[("files", ("set.pdf", b"%PDF-1.7", "application/pdf"))]).json()["id"]


def finished(client, rid):
    for _ in range(400):
        st = client.get(f"/api/reviews/{rid}").json()
        if st["state"] in ("done", "failed"):
            return st
        time.sleep(.02)
    raise AssertionError("review did not finish")


def test_stages_run_after_a_good_ingest(client, monkeypatch):
    stub_ingest(monkeypatch)
    seen = []
    monkeypatch.setattr(pipeline, "STAGES", [stage("read", lambda ctx: seen.append(ctx.sheets) or {"ok": True})])
    st = finished(client, upload(client))
    assert st["state"] == "done" and st["step"] == "Done" and st["sheets"] == 1
    assert [row["state"] for row in st["stages"]] == ["done"]
    assert seen == [SHEETS]


@pytest.mark.parametrize("code, message", [
    ("raise SystemExit(139)", "Could not read this upload"),
    ("import sys, json, pathlib; pathlib.Path(sys.argv[1], 'ingest_result.json').write_text(json.dumps({'error': 'This PDF is password protected'})); raise SystemExit(2)",
     "This PDF is password protected"),
], ids=["crash", "reported"])
def test_failed_ingest_runs_no_stages_and_keeps_its_message(client, monkeypatch, code, message):
    monkeypatch.setattr(server, "INGEST_COMMAND", [sys.executable, "-c", code])
    seen = []
    monkeypatch.setattr(pipeline, "STAGES", [stage("read", lambda ctx: seen.append(1) or {})])
    rid = upload(client)
    st = finished(client, rid)
    assert (st["state"], st["message"]) == ("failed", message)
    assert [row["state"] for row in st["stages"]] == ["queued"] and st["usd"] == 0
    assert seen == []
    assert not (server.DATA / rid / "stage_read.json").exists()


def test_status_lists_the_stages_before_anything_runs(client, monkeypatch):
    monkeypatch.setattr(pipeline, "STAGES", [("read", "Read the sheets", lambda ctx: {})])
    monkeypatch.setattr(server.workers, "submit", lambda *args: None)
    monkeypatch.setattr(server, "waiting", threading.BoundedSemaphore(1))
    assert client.get(f"/api/reviews/{upload(client)}").json() == {
        "state": "queued",
        "step": "Waiting",
        "stages": [{"name": "read", "label": "Read the sheets", "state": "queued", "done": 0, "total": 0}],
        "usd": 0,
    }


def test_review_that_cannot_run_fails_plainly_and_frees_its_slot(client, monkeypatch, caplog):
    stub_ingest(monkeypatch, sheets="not json")
    monkeypatch.setattr(server, "waiting", threading.BoundedSemaphore(1))
    with caplog.at_level(logging.ERROR, logger="storey.pipeline"):
        st = finished(client, upload(client))
    assert (st["state"], st["message"], st["sheets"]) == ("failed", "The review stopped before it finished", 1)
    assert "JSONDecodeError" in caplog.text and "read_json" in caplog.text
    assert "line 1 column 1" not in caplog.text
    assert server.waiting.acquire(timeout=2)
