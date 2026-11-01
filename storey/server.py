"""Web server: accepts a drawing set, runs the review in the background, serves the result and the review screen."""
import hmac
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline
from .assemble import assemble
from .findings import _masked
from .ingest import MAX_SHEETS
from .rules import JURISDICTIONS
from .testtool import remove_stair

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("STOREY_DATA", ROOT / "data/reviews"))
MAX_UPLOAD = 400 * 1024 * 1024
MAX_WAITING = 8       # each review spends API credit, so a burst of uploads must be turned away, not queued
MAX_NAME = 80         # characters of the client's file name kept as a label. It is never a path
SIGNATURES = {".pdf": b"%PDF", ".png": b"\x89PNG", ".jpg": b"\xff\xd8", ".jpeg": b"\xff\xd8"}
INGEST_SECONDS = 900
INGEST_COMMAND = [sys.executable, "-m", "storey.ingest"]
RUN_KEY = secrets.token_urlsafe(24)   # used when no key is configured, and shown once in the terminal
CSP = ("default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
       "img-src 'self' data: blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")

app = FastAPI(title="Storey")
workers = ThreadPoolExecutor(max_workers=2)
waiting = threading.BoundedSemaphore(MAX_WAITING)
edits = threading.Lock()   # a selector change and a Keep click rewrite the same files, and two at once must not lose either


def access_key():
    return os.environ.get("STOREY_ACCESS_KEY") or RUN_KEY


class LimitBody:
    """Stops a request body at the limit while it arrives, so a missing or false Content-Length cannot fill the disk."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        seen = 0

        async def counted():
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > self.limit:
                raise HTTPException(413, "The upload is over 400 MB")
            return message
        await self.app(scope, counted, send)


app.add_middleware(LimitBody, limit=MAX_UPLOAD + 1024 * 1024)


def is_cross_site(request):
    """A page on another site can post a form to localhost without a preflight, so check where it came from."""
    if request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
        return True
    origin = request.headers.get("origin")
    return bool(origin) and urlsplit(origin).netloc != request.headers.get("host")


@app.middleware("http")
async def guard(request: Request, call_next):
    # The API always needs the key. Each review spends credit, and no check of headers or addresses can tell
    # this machine from a proxy with certainty.
    path = request.url.path
    if path.startswith("/api/") and request.method not in ("GET", "HEAD") and is_cross_site(request):
        return JSONResponse({"detail": "Requests from other sites are not allowed"}, 403)
    if path.startswith("/api/") and path != "/api/session":
        given = request.headers.get("x-storey-key") or request.cookies.get("storey_key") or ""
        if not hmac.compare_digest(given.encode(), access_key().encode()):
            return JSONResponse({"detail": "Access key needed"}, 401)
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


class Session(BaseModel):
    key: str


@app.post("/api/session")
def open_session(body: Session, request: Request, response: Response):
    key = access_key()
    if not hmac.compare_digest(body.key.encode(), key.encode()):
        raise HTTPException(401, "That key is not right")
    response.set_cookie("storey_key", key, httponly=True, samesite="strict", secure=request.url.scheme == "https", max_age=7 * 86400)
    return {"ok": True}


def _is_id(name):
    # Ids are server-made hex, so anything else is a bad link or a traversal attempt.
    return len(name) == 32 and not set(name) - set("0123456789abcdef")


def _dir(review_id):
    path = DATA / review_id
    if not _is_id(review_id) or not path.is_dir():
        raise HTTPException(404, "No such review")
    return path


def _finished(review_id):
    path = _dir(review_id)
    if pipeline.read_json(path / "status.json").get("state") != "done":
        raise HTTPException(409, "This review has not finished")
    return path


def _status(path, **fields):
    pipeline.write_json(path / "status.json", pipeline.read_json(path / "status.json") | fields)


def _ingest(path, uploads):
    r = subprocess.run([*INGEST_COMMAND, str(path), *map(str, uploads)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=INGEST_SECONDS)
    try:
        out = json.loads((path / "ingest_result.json").read_text())
    except (OSError, ValueError):
        out = {}
    if r.returncode or "sheets" not in out:
        raise ValueError(out.get("error", "Could not read this upload"))
    return out["sheets"]


def _stages(path):
    try:
        pipeline.run(path)
    except Exception as e:
        pipeline.log.error("Review %s stopped with %s\n%s", path.name, type(e).__name__, "".join(traceback.format_tb(e.__traceback__)))
        _status(path, state="failed", step="Stopped", message="The review stopped before it finished")


def _run(path, uploads):
    try:
        _status(path, state="running", step="Reading the drawings")
        started = time.monotonic()
        _status(path, sheets=_ingest(path, uploads))
        pipeline.write_json(path / "cost.json", {"ingest": {"usd": 0, "seconds": round(time.monotonic() - started, 1)}})
    except subprocess.TimeoutExpired:
        _status(path, state="failed", step="Stopped", message="This upload took too long to read")
    except ValueError as e:
        _status(path, state="failed", step="Stopped", message=str(e))
    except Exception:
        _status(path, state="failed", step="Stopped", message="Could not read this upload")
    else:
        _stages(path)
    finally:
        waiting.release()


def _run_copy(path):
    try:
        _stages(path)
    finally:
        waiting.release()


async def _save(upload, dest):
    """Writes one file to disk in chunks and returns its size. The first bytes must match the file type."""
    ext = dest.suffix
    head = await upload.read(8)
    if not head.startswith(SIGNATURES[ext]):
        raise HTTPException(415, f"{upload.filename} is not a PDF, PNG or JPEG file")
    size = len(head)
    with open(dest, "wb") as out:
        out.write(head)
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "The upload is over 400 MB")
            out.write(chunk)
    return size


async def _accept(files, parent=None):
    if not files:
        raise HTTPException(400, "Choose a PDF or page images to upload")
    if len(files) > MAX_SHEETS:
        raise HTTPException(413, f"Upload at most {MAX_SHEETS} files")
    if not waiting.acquire(blocking=False):
        raise HTTPException(429, "Too many reviews are running. Try again in a few minutes.")
    path = DATA / uuid.uuid4().hex
    try:
        (path / "upload").mkdir(parents=True)
        saved, total = [], 0
        for i, f in enumerate(files):
            ext = Path(f.filename or "").suffix.lower()
            if ext not in SIGNATURES:
                raise HTTPException(415, f"{f.filename} is not a PDF, PNG or JPEG file")
            p = path / "upload" / f"{i:04d}{ext}"   # never trust the client's file name on disk
            total += await _save(f, p)
            if total > MAX_UPLOAD:
                raise HTTPException(413, "The upload is over 400 MB")
            saved.append(p)
        name = Path(files[0].filename or "").stem[:MAX_NAME] if len(files) == 1 else f"{len(files)} page images"
        pipeline.write_json(path / "stage_upload.json", {"name": _masked(name), "parent": parent})   # a file can be called Approved, and the list of reviews shows this name
        _status(path, state="queued", step="Waiting", stages=pipeline.stage_rows(pipeline.STAGES), usd=0)
    except BaseException:
        waiting.release()
        shutil.rmtree(path, ignore_errors=True)
        raise
    workers.submit(_run, path, saved)
    return {"id": path.name}


@app.post("/api/reviews")
async def create_review(files: list[UploadFile]):
    return await _accept(files)


@app.get("/api/reviews")
def past_reviews():
    rows = []
    for path in DATA.iterdir() if DATA.is_dir() else ():
        st = pipeline.read_json(path / "status.json") if _is_id(path.name) else None
        if st:
            upload = pipeline.read_json(path / "stage_upload.json")
            rows.append({"id": path.name, "name": upload.get("name", ""), "parent": upload.get("parent"), "state": st.get("state"), "sheets": st.get("sheets"), "usd": st.get("usd", 0), "updated": (path / "status.json").stat().st_mtime})
    return sorted(rows, key=lambda row: row["updated"], reverse=True)


@app.get("/api/reviews/{review_id}")
def review_status(review_id: str):
    return json.loads((_dir(review_id) / "status.json").read_text())


@app.get("/api/reviews/{review_id}/files/{name:path}")
def review_file(review_id: str, name: str):
    base = _dir(review_id).resolve()
    target = (base / name).resolve()
    if target.parent not in (base, base / "verticals") or not target.is_file():   # upload/ holds the file as it arrived, and no screen needs it
        raise HTTPException(404, "No such file")
    return FileResponse(target)


class Evaluate(BaseModel):
    jurisdiction: Literal[*JURISDICTIONS] = "houston"
    sprinklers: Literal["yes", "no"] | None = None
    discharge: int | None = None


@app.post("/api/reviews/{review_id}/evaluate")
def reevaluate(review_id: str, body: Evaluate):
    path = _finished(review_id)
    if body.discharge is not None and body.discharge not in pipeline.read_json(path / "stage_floors.json").get("order", ()):
        raise HTTPException(422, "That sheet is not in the floor order")
    with edits:
        return assemble(path, body.jurisdiction, body.sprinklers, body.discharge)


class Decision(BaseModel):
    key: str
    decision: Literal["keep", "drop"] | None


@app.post("/api/reviews/{review_id}/decisions")
def decide(review_id: str, body: Decision):
    path = _finished(review_id)
    with edits:
        evaluated = pipeline.read_json(path / "stage_evaluate.json")
        if body.key not in {f["key"] for f in evaluated.get("findings", ())}:
            raise HTTPException(404, "No such finding")
        pipeline.write_json(path / "decisions.json", pipeline.read_json(path / "decisions.json") | {body.key: body.decision})
        return assemble(path, **evaluated["choices"])


@app.post("/api/reviews/{review_id}/recheck")
async def recheck(review_id: str, files: list[UploadFile]):
    _finished(review_id)
    return await _accept(files, review_id)


class Removal(BaseModel):
    vertical: str
    sheet: int


@app.post("/api/reviews/{review_id}/remove")
def remove(review_id: str, body: Removal):
    """Starts a labelled test copy of a finished review with one stair removed. The stage run spends credit, so it takes a slot like an upload."""
    path = _finished(review_id)
    if not waiting.acquire(blocking=False):
        raise HTTPException(429, "Too many reviews are running. Try again in a few minutes.")
    try:
        with edits:   # the copy must not catch a selector change half written
            copy = remove_stair(path, body.vertical, body.sheet)
    except ValueError as e:
        waiting.release()
        raise HTTPException(422, str(e)) from None
    except BaseException:
        waiting.release()
        raise
    workers.submit(_run_copy, copy)
    return {"id": copy.name}


app.mount("/", StaticFiles(directory=ROOT / "app/web", html=True), name="web")
