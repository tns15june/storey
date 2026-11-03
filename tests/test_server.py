import threading
import time

import cv2
import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient


KEY = {"x-storey-key": "test-key"}
ID_URL = "/api/reviews/" + "0" * 32


@pytest.fixture
def client(tmp_path, monkeypatch):
    from storey import meta, pipeline, server
    monkeypatch.setattr(server, "DATA", tmp_path)
    monkeypatch.setattr(pipeline, "STAGES", [])   # these tests are about the server, and the stages would call the model
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("a test reached the real client"))
    monkeypatch.setenv("STOREY_ACCESS_KEY", "test-key")
    yield TestClient(server.app, base_url="http://127.0.0.1", headers=KEY)
    idle = threading.Barrier(server.workers._max_workers)   # every worker free, so no upload outlives the stubs above and reaches the real stages
    for done in [server.workers.submit(idle.wait) for _ in range(idle.parties)]:
        done.result(timeout=60)


def _pdf_bytes():
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 60), "A-101")
    return doc.tobytes()


def _wait(client, rid):
    for _ in range(400):
        st = client.get(f"/api/reviews/{rid}").json()
        if st["state"] in ("done", "failed"):
            return st
        time.sleep(.02)
    raise AssertionError("review did not finish")


def test_upload_then_fetch_a_sheet(client):
    r = client.post("/api/reviews", files=[("files", ("set.pdf", _pdf_bytes(), "application/pdf"))])
    assert r.status_code == 200
    rid = r.json()["id"]
    assert _wait(client, rid)["sheets"] == 1
    assert client.get(f"/api/reviews/{rid}/files/sheet_000.png").status_code == 200


def _jpg():
    return cv2.imencode(".jpg", np.full((40, 60, 3), 255, np.uint8))[1].tobytes()


def test_upload_takes_several_images_as_one_set(client):
    r = client.post("/api/reviews", files=[("files", (f"{i:02d}.jpg", _jpg(), "image/jpeg")) for i in range(5)])
    assert r.status_code == 200
    st = _wait(client, r.json()["id"])
    assert (st["state"], st["sheets"]) == ("done", 5)


def test_rejects_a_file_that_is_not_what_it_claims(client):
    r = client.post("/api/reviews", files=[("files", ("set.pdf", b"MZ not a pdf", "application/pdf"))])
    assert r.status_code == 415


@pytest.mark.parametrize("name", ["..%2F..%2Fsecret", "status.json%2F..%2F..%2Fx"])
def test_file_route_blocks_traversal(client, name):
    r = client.post("/api/reviews", files=[("files", ("set.pdf", _pdf_bytes(), "application/pdf"))])
    rid = r.json()["id"]
    _wait(client, rid)
    assert client.get(f"/api/reviews/{rid}/files/{name}").status_code == 404


def test_unknown_review_is_404(client):
    assert client.get("/api/reviews/" + "0" * 32).status_code == 404
    assert client.get("/api/reviews/not-an-id").status_code == 404


def test_rejected_upload_leaves_nothing_on_disk(client, tmp_path):
    client.post("/api/reviews", files=[("files", ("set.pdf", b"MZ not a pdf", "application/pdf"))])
    assert list(tmp_path.iterdir()) == []


def test_oversize_body_is_stopped_while_it_arrives(client, monkeypatch):
    from storey import server
    monkeypatch.setattr(server, "MAX_UPLOAD", 10)
    body = b"%PDF" + b"x" * 100
    r = client.post("/api/reviews", files=[("files", ("set.pdf", body, "application/pdf"))])
    assert r.status_code == 413


def test_busy_server_turns_uploads_away_without_leaking_slots(client, monkeypatch):
    import threading
    from storey import server
    monkeypatch.setattr(server, "waiting", threading.BoundedSemaphore(1))
    server.waiting.acquire()
    r = client.post("/api/reviews", files=[("files", ("set.pdf", _pdf_bytes(), "application/pdf"))])
    assert r.status_code == 429
    server.waiting.release()
    r = client.post("/api/reviews", files=[("files", ("set.pdf", b"nope", "application/pdf"))])
    assert r.status_code == 415
    assert server.waiting.acquire(blocking=False)


def test_api_always_needs_the_key(tmp_path, monkeypatch):
    from storey import server
    monkeypatch.setattr(server, "DATA", tmp_path)
    monkeypatch.delenv("STOREY_ACCESS_KEY", raising=False)
    local = TestClient(server.app, base_url="http://localhost", client=("127.0.0.1", 50000))
    assert local.get(ID_URL).status_code == 401
    assert local.post("/api/reviews", files=[("files", ("set.pdf", _pdf_bytes(), "application/pdf"))]).status_code == 401
    assert local.get(ID_URL, headers={"x-storey-key": server.RUN_KEY}).status_code == 404


@pytest.mark.parametrize("headers", [{"origin": "https://evil.example"}, {"sec-fetch-site": "cross-site"}])
def test_other_sites_cannot_post_to_a_local_server(client, headers):
    r = client.post("/api/reviews", headers=headers, files=[("files", ("set.pdf", _pdf_bytes(), "application/pdf"))])
    assert r.status_code == 403


def test_same_origin_browser_post_is_allowed(client):
    headers = {"origin": "http://127.0.0.1", "sec-fetch-site": "same-origin"}
    r = client.post("/api/reviews", headers=headers, files=[("files", ("set.pdf", _pdf_bytes(), "application/pdf"))])
    assert r.status_code == 200


@pytest.mark.parametrize("state, after", [("queued", "failed"), ("running", "failed"), ("done", "done")])
def test_a_restart_closes_reviews_left_running(client, tmp_path, state, after):
    from storey import pipeline, server
    path = tmp_path / ("a" * 32)
    path.mkdir()
    pipeline.write_json(path / "status.json", {"state": state, "stages": []})
    server.close_interrupted()
    got = client.get(f"/api/reviews/{path.name}").json()
    assert got["state"] == after and ("restarted" in got.get("message", "")) == (after == "failed")


def test_a_hung_reader_is_killed_and_frees_its_slot(client, monkeypatch):
    import sys
    from storey import server
    monkeypatch.setattr(server, "INGEST_COMMAND", [sys.executable, "-c", "import time; time.sleep(30)"])
    monkeypatch.setattr(server, "INGEST_SECONDS", .2)
    rid = client.post("/api/reviews", files=[("files", ("set.pdf", _pdf_bytes(), "application/pdf"))]).json()["id"]
    st = _wait(client, rid)
    assert st["state"] == "failed" and "too long" in st["message"]


def test_a_crashing_reader_fails_cleanly(client, monkeypatch):
    import sys
    from storey import server
    monkeypatch.setattr(server, "INGEST_COMMAND", [sys.executable, "-c", "raise SystemExit(139)"])
    rid = client.post("/api/reviews", files=[("files", ("set.pdf", _pdf_bytes(), "application/pdf"))]).json()["id"]
    st = _wait(client, rid)
    assert st["state"] == "failed" and st["message"] == "Could not read this upload"


def test_session_cookie_works_in_place_of_the_header(tmp_path, monkeypatch):
    from storey import server
    monkeypatch.setattr(server, "DATA", tmp_path)
    monkeypatch.setenv("STOREY_ACCESS_KEY", "s3cret")
    browser = TestClient(server.app, base_url="http://127.0.0.1")
    assert browser.get(ID_URL).status_code == 401
    assert browser.post("/api/session", json={"key": "wrong"}).status_code == 401
    assert browser.post("/api/session", json={"key": "s3cret"}).status_code == 200
    assert browser.get(ID_URL).status_code == 404


NEW_ROUTES = [("post", "/evaluate", {"json": {}}), ("post", "/decisions", {"json": {"key": "x", "decision": "keep"}}),
              ("post", "/recheck", {"files": [("files", ("set.pdf", b"%PDF-1.7", "application/pdf"))]})]


@pytest.mark.parametrize("method, route, body", NEW_ROUTES)
def test_key_required_on_new_routes(tmp_path, monkeypatch, method, route, body):
    from storey import server
    monkeypatch.setattr(server, "DATA", tmp_path)
    monkeypatch.setenv("STOREY_ACCESS_KEY", "test-key")
    assert getattr(TestClient(server.app, base_url="http://127.0.0.1"), method)(ID_URL + route, **body).status_code == 401


def test_key_required_on_the_list_of_reviews(tmp_path, monkeypatch):
    from storey import server
    monkeypatch.setattr(server, "DATA", tmp_path)
    monkeypatch.setenv("STOREY_ACCESS_KEY", "test-key")
    assert TestClient(server.app, base_url="http://127.0.0.1").get("/api/reviews").status_code == 401


@pytest.mark.parametrize("method, route, body", NEW_ROUTES)
@pytest.mark.parametrize("headers", [{"origin": "https://evil.example"}, {"sec-fetch-site": "cross-site"}])
def test_cross_site_post_refused_on_new_routes(client, method, route, body, headers):
    assert getattr(client, method)(ID_URL + route, headers=headers, **body).status_code == 403


@pytest.mark.parametrize("method, route, body", NEW_ROUTES)
def test_new_routes_refuse_a_bad_id(client, method, route, body):
    assert getattr(client, method)("/api/reviews/not-an-id" + route, **body).status_code == 404


def test_the_review_screen_itself_needs_no_key():
    from storey import server
    assert TestClient(server.app, base_url="http://127.0.0.1").get("/").status_code == 200


def test_responses_carry_a_content_policy(client):
    assert "default-src 'self'" in client.get(ID_URL).headers["content-security-policy"]


def test_reader_noise_on_stdout_is_ignored(client, monkeypatch):
    import sys
    from storey import server
    flood = "import sys, json, pathlib; print('warning' * 1000); pathlib.Path(sys.argv[1], 'ingest_result.json').write_text(json.dumps({'sheets': 3}))"
    monkeypatch.setattr(server, "INGEST_COMMAND", [sys.executable, "-c", flood])
    rid = client.post("/api/reviews", files=[("files", ("set.pdf", _pdf_bytes(), "application/pdf"))]).json()["id"]
    assert _wait(client, rid)["sheets"] == 3


WROTE = "import sys, pathlib, time; d = pathlib.Path(sys.argv[1]); (d / 'sheet_000.png').write_bytes(b'x' * 1000); (d / 'wrote').touch(); "
READERS = {"killed at its time limit": WROTE + "time.sleep(30)", "crashed": WROTE + "raise SystemExit(139)", "garbled result": WROTE + "(d / 'ingest_result.json').write_text('[]')"}


@pytest.mark.parametrize("reader", READERS.values(), ids=READERS.keys())
def test_a_reader_that_fails_leaves_no_sheets(client, monkeypatch, tmp_path, reader):
    import sys
    from storey import server
    monkeypatch.setattr(server, "INGEST_COMMAND", [sys.executable, "-c", reader])
    monkeypatch.setattr(server, "INGEST_SECONDS", 1)
    rid = client.post("/api/reviews", files=[("files", ("set.pdf", _pdf_bytes(), "application/pdf"))]).json()["id"]
    assert _wait(client, rid)["state"] == "failed"
    assert (tmp_path / rid / "wrote").exists()   # the reader did render a sheet before it stopped
    assert not list((tmp_path / rid).glob("sheet_*.png"))
