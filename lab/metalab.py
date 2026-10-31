"""Experiment helper: Meta Model API calls with a spend ledger and a hard budget stop.

Set LAB_NAME to tag calls. LAB_BUDGET caps that name's spend in USD, and LAB_TOTAL caps the shared ledger.
The key is read from .env inside this process, so scripts never need to see it. Never print it.
"""
import base64
import fcntl
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format, video_segmentation_format
from openai import APIConnectionError, APIStatusError, OpenAI

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "data/lab/ledger.jsonl"
BASE = "https://api.meta.ai/v1"
PRICES = {"muse-wonder-1.0": (4.0, 0.20, 20.0), "muse-spark-1.3": (1.25, 0.15, 4.25), "muse-spark-1.2": (1.25, 0.15, 4.25), "muse-spark-1.1": (1.25, 0.15, 4.25)}
NAME = os.environ.get("LAB_NAME", "untagged")
BUDGET = float(os.environ.get("LAB_BUDGET", "2"))
TOTAL = float(os.environ.get("LAB_TOTAL", "60"))


class BudgetExceeded(RuntimeError):
    pass


def _key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("MODEL_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("MODEL_API_KEY is missing from .env")


client = OpenAI(base_url=BASE, api_key=_key(), timeout=600, max_retries=0)


def spent(name=None):
    if not LEDGER.exists():
        return 0.0
    rows = (json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip())
    return sum(r["usd"] for r in rows if name is None or r["name"] == name)


def _guard():
    if spent() >= TOTAL or spent(NAME) >= BUDGET:
        raise BudgetExceeded(f"{NAME}: {spent(NAME):.3f} of {BUDGET}, ledger {spent():.3f} of {TOTAL}")


def _log(model, usd, **extra):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"t": time.time(), "name": NAME, "model": model, "usd": round(usd, 6), **extra}) + "\n")


def _retry(fn):
    for attempt in range(6):
        try:
            return fn()
        except (APIConnectionError, APIStatusError) as e:
            code = getattr(e, "status_code", 0)
            if attempt == 5 or (code and code not in (408, 409, 429) and code < 500):
                raise
            time.sleep(min(30, 0.5 * 2 ** attempt) + random.random())


def data_url(im, quality=92):
    """A data URL for a path or a BGR array. PNG files pass through unchanged."""
    if isinstance(im, (str, Path)):
        p = Path(im)
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[p.suffix.lower()]
        return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]).decode()


def spark(prompt, images=(), model="muse-spark-1.3", schema=None, effort=None, system=None, max_tokens=24000):
    """One chat call. Returns (parsed JSON or text, info with usd, seconds and token counts)."""
    _guard()
    content = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": data_url(i)}} for i in images]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    kw = {}
    if schema:
        kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}}
    if effort:
        kw["reasoning_effort"] = effort
    # Always capped: two uncapped calls ran to 262,144 output tokens at 1.12 USD each.
    kw["max_completion_tokens"] = max_tokens
    t0 = time.time()
    r = _retry(lambda: client.chat.completions.create(model=model, messages=msgs, **kw))
    u = r.usage
    cached = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
    pi, pc, po = PRICES[model]
    usd = ((u.prompt_tokens - cached) * pi + cached * pc + u.completion_tokens * po) / 1e6
    _log(model, usd, input=u.prompt_tokens, output=u.completion_tokens, images=len(images))
    text = r.choices[0].message.content
    return (json.loads(text) if schema else text), {"usd": usd, "seconds": time.time() - t0, "input": u.prompt_tokens, "output": u.completion_tokens}


def _sam(prompt, media, metadata):
    _guard()

    def go():
        stream = client.responses.create(model="sam-3.1", stream=True, metadata=metadata,
                                         input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}, media]}])
        text = ""
        for ev in stream:
            ev = ev.model_dump()
            if ev.get("type") == "response.output_text.delta":
                text += ev["delta"]
            elif ev.get("type") in ("response.failed", "response.incomplete", "error"):
                raise RuntimeError(f"SAM {ev.get('type')}: {json.dumps(ev, default=str)[:400]}")
        return text
    return _retry(go)


def _masks(text, fmt):
    parser = fmt().create_parser()
    parser.push(text if text.endswith("\n") else text + "\n")
    return [r for r in parser.finish(CompletedOutcome()).result.records if r.kind == "mask"]


def raster(rec):
    """A record's mask as a bool array the size of its box."""
    b, m = rec.bounds, rec.mask
    r = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
    if (m.width, m.height) != (b.right - b.left, b.bottom - b.top):
        r = cv2.resize(r, (b.right - b.left, b.bottom - b.top), interpolation=cv2.INTER_NEAREST)
    return r.astype(bool)


def sam_image(prompt, im):
    """Concept text or a box and point prompt on one image. Returns mask records."""
    t0 = time.time()
    text = _sam(prompt, {"type": "input_image", "image_url": data_url(im)}, {"include_confidence": "true", "mask_encoding": "one_bit"})
    _log("sam-3.1", 0.0025, kind="image", seconds=round(time.time() - t0, 2))
    return _masks(text, image_segmentation_format)


def sam_video(prompt, path, frames, direction="both"):
    """Box or point prompts on an MP4. `frames` is the clip length, used for the spend estimate."""
    t0 = time.time()
    # Bytes, not an open handle: a retried upload would otherwise send a spent file object.
    video = (Path(path).name, Path(path).read_bytes())
    fid = _retry(lambda: client.files.create(purpose="user_data", file=video)).id
    text = _sam(prompt, {"type": "input_video", "file_id": fid}, {"include_confidence": "true", "propagation_direction": direction, "mask_encoding": "one_bit"})
    _log("sam-3.1", max(frames, 50) * 0.0002, kind="video", frames=frames, seconds=round(time.time() - t0, 2))
    return _masks(text, video_segmentation_format)


if __name__ == "__main__":
    # Self-check with no API call: the spend arithmetic and the guard.
    assert abs(sum(PRICES["muse-spark-1.3"]) - 5.65) < 1e-9
    print("ledger total", round(spent(), 4))
