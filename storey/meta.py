"""Meta Model API client. Every model call goes through here, so each one is cached and metered, and each Spark call is token-capped.
SAM is billed per image or frame, so it takes no token cap.

The key is loaded in this module only, on the first real call, and is never logged.
"""
import base64
import fcntl
import functools
import hashlib
import json
import os
import random
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format, video_segmentation_format
from openai import APIConnectionError, APIStatusError, OpenAI

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("STOREY_DATA", ROOT / "data/reviews"))
CACHE = Path(os.environ.get("STOREY_CACHE", ROOT / "data/cache"))
BASE = "https://api.meta.ai/v1"
SAM = "sam-3.1"
PRICES = {"muse-spark-1.3": (1.25, 0.15, 4.25), "muse-wonder-1.0": (4.0, 0.20, 20.0)}   # USD per million input, cached, output tokens
SAM_IMAGE_USD = 0.0025
SAM_FRAME_USD = 0.0002
SAM_MIN_FRAMES = 50
SAM_BAD_EVENTS = ("response.failed", "response.incomplete", "response.refusal.delta", "response.refusal.done", "error")


class BudgetReached(RuntimeError):
    pass


class _Retryable(RuntimeError):
    """A stream failure that Meta's docs say to replay in full."""


class Meter:
    """Spend for one review. The ledger is shared by every review, so it also carries the server's daily total."""

    def __init__(self, cap_usd: float = 8.0, daily_cap_usd: float = 25.0, ledger: Path = DATA / "ledger.jsonl"):
        self.cap_usd, self.daily_cap_usd, self.ledger = cap_usd, daily_cap_usd, ledger
        self._spent = 0.0
        self._lock = threading.Lock()

    def add(self, usd: float, **fields) -> None:
        with self._lock:
            self._spent += usd
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps({"t": time.time(), "usd": round(usd, 6), **fields}) + "\n")

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    def check(self) -> None:
        # shortcut: parallel calls can all pass here before any adds so overshoot can reach one call per thread, reserve an estimate here if a cap must hold tighter.
        if self.spent >= self.cap_usd:
            raise BudgetReached(f"This review reached its spend cap of {self.cap_usd:.2f} USD.")
        if self._spent_today() >= self.daily_cap_usd:
            raise BudgetReached(f"The server reached its daily spend cap of {self.daily_cap_usd:.2f} USD.")

    def _spent_today(self):
        try:
            with open(self.ledger) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                lines = f.readlines()
        except FileNotFoundError:
            return 0.0
        today, total = time.time() // 86400, 0.0   # whole days of epoch seconds are UTC days
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue   # one torn write must not stop every later call
            if row["t"] // 86400 == today:
                total += row["usd"]
        return total


def _api_key():
    key = os.environ.get("MODEL_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else ():
        if line.startswith("MODEL_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("MODEL_API_KEY is not set in the environment or in .env")


@functools.cache
def _client():
    return OpenAI(base_url=BASE, api_key=_api_key(), timeout=600, max_retries=0)


def _http(kind, **request):
    client = _client()
    if kind == "upload":
        return client.files.create(**request).id
    if kind == "chat":
        r = client.chat.completions.create(**request)
        reply = {"text": None}
        if r.choices:
            reply |= {"text": r.choices[0].message.content, "finish": r.choices[0].finish_reason}
        if r.usage:
            details = r.usage.prompt_tokens_details
            reply |= {"input": r.usage.prompt_tokens, "cached": (details.cached_tokens or 0) if details else 0, "output": r.usage.completion_tokens}
        return reply
    text, completed = "", False
    for ev in client.responses.create(stream=True, **request):
        ev = ev.model_dump()
        part = ev.get("part") or {}
        if ev.get("type") == "response.output_text.delta":
            text += ev["delta"]
        elif ev.get("type") == "response.content_part.done" and part.get("type") == "output_text":
            text = text or part["text"]   # with deltas, this part repeats them
        elif ev.get("type") == "response.completed":
            completed = True
        elif ev.get("type") == "error" and ev.get("code") == "server_shutting_down":
            raise _Retryable("The SAM server shut down during the stream")
        elif ev.get("type") in SAM_BAD_EVENTS or part.get("type") == "refusal":
            raise RuntimeError(f"SAM {ev.get('type')}: {json.dumps(ev, default=str)[:400]}")
    if not completed:
        raise RuntimeError("The SAM stream ended before it completed")   # partial output must not reach the cache
    return text


_transport = _http


def set_transport(fake) -> None:
    """Test seam. `fake(kind, **request)` replaces the HTTP call, and None restores it. Replies are plain data:

    kind "chat", request is the chat.completions body: return {"text": str | None}, with optional "input", "cached"
    and "output" token counts and "finish" (the finish_reason).
    kind "sam", request is the responses body: return the SAM output text.
    kind "upload", request holds purpose and file as (name, bytes): return the file id.
    """
    global _transport
    _transport = _http if fake is None else fake


def _retry(fn):
    for attempt in range(6):
        try:
            return fn()
        except (APIConnectionError, APIStatusError, _Retryable) as e:
            code = getattr(e, "status_code", 0)
            # A 504 is not retried because Meta's docs say the same request usually times out again.
            if attempt == 5 or code == 504 or (code and code not in (408, 409, 429) and code < 500):
                raise
            # shortcut: ignores Retry-After so a 503 asking for 60 s can use up the retries first, read the header if overload stops reviews.
            time.sleep(min(30, 0.5 * 2 ** attempt) + random.random())


def _encode(image):
    """(mime, bytes) for a path or a BGR array. Files pass through unchanged, and arrays become PNG so no pixel changes."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[path.suffix.lower()]
        return mime, path.read_bytes()
    return "image/png", cv2.imencode(".png", image)[1].tobytes()


def _data_url(mime, blob):
    return f"data:{mime};base64," + base64.b64encode(blob).decode()


def _cache_key(model, mode, system, prompt, schema, blobs):
    """`mode` is the effort on Spark, and the direction or "image" on SAM."""
    h = hashlib.sha256(json.dumps([model, mode, system, prompt, schema], sort_keys=True).encode())
    for blob in blobs:
        h.update(hashlib.sha256(blob).digest())
    return h.hexdigest()


def _cached(key):
    try:
        return (CACHE / key).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _store(key, text):
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"{key}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(CACHE / key)   # a thread reading this key must never see half a file


def _parsed(text, schema):
    if not text or not text.strip():
        raise ValueError("The model returned an empty reply")
    if not schema:
        return text
    out = json.loads(text)
    if not isinstance(out, dict):
        raise ValueError("The model's reply is not a JSON object")
    return out


def spark(prompt: str, images: Sequence = (), *, meter: Meter, schema: dict | None = None, effort: str = "low", model: str = "muse-spark-1.3",
          system: str | None = None, max_tokens: int = 24000, fresh: bool = False, run: int = 0) -> dict | str:
    """One chat call: the parsed object with a schema, otherwise the text. A malformed, empty or cut-off reply raises ValueError.
    `run` repeats a call: a non-zero run has its own cache entry, the model never sees it, and without it identical requests collapse into one reply."""
    input_price, cached_price, output_price = PRICES[model]   # a model not listed, such as a Contributor variant, stops here
    encoded = [_encode(image) for image in images]
    key = _cache_key(model, f"{effort}#{run}" if run else effort, system, prompt, schema, [blob for _, blob in encoded])
    text = None if fresh else _cached(key)
    if text is not None:
        return _parsed(text, schema)
    content = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": _data_url(*e)}} for e in encoded]
    request = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}],
        "reasoning_effort": effort,
        "max_completion_tokens": max_tokens,   # always capped: two uncapped calls ran to 262,144 output tokens at 1.12 USD each
    }
    if schema:
        request["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}}
    for attempt in range(2):
        meter.check()
        reply = _retry(lambda: _transport("chat", **request))
        tokens_in, cached, tokens_out = reply.get("input", 0), reply.get("cached", 0), reply.get("output", 0)
        usd = ((tokens_in - cached) * input_price + cached * cached_price + tokens_out * output_price) / 1e6
        meter.add(usd, model=model, input=tokens_in, output=tokens_out, images=len(encoded))
        finish = reply.get("finish")
        if finish in ("length", "content_filter"):   # the same request would be cut off again
            raise ValueError(f"The model's reply was cut off ({finish})")
        try:
            out = _parsed(reply["text"], schema)
        except ValueError:
            if attempt:
                raise
            continue
        _store(key, reply["text"])
        return out


def _masks(text, fmt):
    parser = fmt().create_parser()
    parser.push(text)
    result = parser.finish(CompletedOutcome()).result
    masks = [r for r in result.records if r.kind == "mask"]
    # Empty output means zero matches, which is valid. Any other output with no masks, or a rejected line, is not.
    if any(d.severity == "error" for d in result.diagnostics) or (text.strip() and not masks):
        raise ValueError("SAM returned output that could not be read")
    return masks


def _raster(rec):
    """A record's mask as a bool array the size of its box."""
    b, m = rec.bounds, rec.mask
    r = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
    if (m.width, m.height) != (b.right - b.left, b.bottom - b.top):
        r = cv2.resize(r, (b.right - b.left, b.bottom - b.top), interpolation=cv2.INTER_NEAREST)
    return r.astype(bool)


def _box(rec):
    b = rec.bounds
    return [b.left, b.top, b.right, b.bottom]


def _sam(prompt, media, metadata, fmt, key, usd, meter, **fields):
    """The mask records of one SAM call. `media` is a function so that a cache hit skips the upload."""
    # shortcut: SAM has no fresh flag so a live demo reuses cached masks, add one if a demo must rerun SAM.
    text = _cached(key)
    if text is not None:
        return _masks(text, fmt)
    meter.check()
    content = [{"type": "input_text", "text": prompt}, media()]
    request = {
        "model": SAM,
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit", **metadata},
        "input": [{"type": "message", "role": "user", "content": content}],
    }
    text = _retry(lambda: _transport("sam", **request))
    meter.add(usd, model=SAM, **fields)
    masks = _masks(text, fmt)
    _store(key, text)
    return masks


def sam_image(prompt: str, image, *, meter: Meter) -> list[dict]:
    """Concept text or a box and point prompt on one image. `conf` is None if SAM left it out."""
    mime, blob = _encode(image)
    media = {"type": "input_image", "image_url": _data_url(mime, blob)}
    key = _cache_key(SAM, "image", None, prompt, None, [blob])
    records = _sam(prompt, lambda: media, {}, image_segmentation_format, key, SAM_IMAGE_USD, meter, kind="image")
    return [{"object": int(r.object_id), "box": _box(r), "conf": r.confidence, "mask": _raster(r)} for r in records]


def sam_video(prompt: str, path: Path, frames: int, *, meter: Meter, direction: str = "both") -> list[dict]:
    """Box or point prompts on an MP4. `frames` is the clip length, used for the cost. `conf` is None if SAM left it out. `mask` is a bool array the size of `box`."""
    name, blob = Path(path).name, Path(path).read_bytes()   # bytes, not an open handle: a retried upload would otherwise send a spent file object

    def media():
        return {"type": "input_video", "file_id": _retry(lambda: _transport("upload", purpose="user_data", file=(name, blob)))}

    key = _cache_key(SAM, direction, None, prompt, None, [blob])
    usd = max(frames, SAM_MIN_FRAMES) * SAM_FRAME_USD
    records = _sam(prompt, media, {"propagation_direction": direction}, video_segmentation_format, key, usd, meter, kind="video", frames=frames)
    return [{"object": int(r.object_id), "frame": r.frame.frame_index, "box": _box(r), "conf": r.confidence, "mask": _raster(r)} for r in records]
