"""Web server: accepts a drawing set, runs the review in the background, serves the result and the review screen."""
import json
import os
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ingest import ingest

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("STOREY_DATA", ROOT / "data/reviews"))
MAX_UPLOAD = 400 * 1024 * 1024
SIGNATURES = {".pdf": b"%PDF", ".png": b"\x89PNG", ".jpg": b"\xff\xd8", ".jpeg": b"\xff\xd8"}

app = FastAPI(title="Storey")


def _dir(review_id):
    # Ids are server-made hex, so anything else is a bad link or a traversal attempt.
    if not review_id.isalnum() or len(review_id) != 32:
        raise HTTPException(404, "No such review")
    path = DATA / review_id
    if not path.is_dir():
        raise HTTPException(404, "No such review")
    return path


def _status(path, **fields):
    tmp = path / "status.tmp"
    tmp.write_text(json.dumps(fields))
    tmp.replace(path / "status.json")   # a poller must never read a half-written file


def _run(path, uploads):
    try:
        _status(path, state="running", step="Reading the drawings")
        sheets = ingest(uploads, path)
        _status(path, state="done", step="Done", sheets=len(sheets))
    except Exception as e:
        _status(path, state="failed", step="Stopped", message=str(e))


@app.post("/api/reviews")
async def create_review(files: list[UploadFile]):
    if not files:
        raise HTTPException(400, "Choose a PDF or page images to upload")
    path = DATA / uuid.uuid4().hex
    (path / "upload").mkdir(parents=True)
    saved, total = [], 0
    for i, f in enumerate(files):
        ext = Path(f.filename or "").suffix.lower()
        body = await f.read()
        total += len(body)
        if ext not in SIGNATURES or not body.startswith(SIGNATURES[ext]):
            raise HTTPException(415, f"{f.filename} is not a PDF, PNG or JPEG file")
        if total > MAX_UPLOAD:
            raise HTTPException(413, "The upload is over 400 MB")
        p = path / "upload" / f"{i:04d}{ext}"   # never trust the client's file name on disk
        p.write_bytes(body)
        saved.append(p)
    _status(path, state="queued", step="Waiting")
    threading.Thread(target=_run, args=(path, saved), daemon=True).start()
    return {"id": path.name}


@app.get("/api/reviews/{review_id}")
def review_status(review_id: str):
    return json.loads((_dir(review_id) / "status.json").read_text())


@app.get("/api/reviews/{review_id}/files/{name}")
def review_file(review_id: str, name: str):
    target = (_dir(review_id) / name).resolve()
    if target.parent != (DATA / review_id).resolve() or not target.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(target)


app.mount("/", StaticFiles(directory=ROOT / "app/web", html=True), name="web")
