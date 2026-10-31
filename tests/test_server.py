import time

import pymupdf
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from storey import server
    monkeypatch.setattr(server, "DATA", tmp_path)
    return TestClient(server.app)


def _pdf_bytes():
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 60), "A-101")
    return doc.tobytes()


def _wait(client, rid):
    for _ in range(100):
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
