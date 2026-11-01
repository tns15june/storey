"""samtricks helpers on top of metalab: Spark with free part order or video, cached SAM concept calls, box scoring. Every call passes metalab's guard and ledger."""
import base64
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "lab")
sys.path.insert(0, "lab/samtricks")
import metalab as m  # noqa: E402
import reg  # noqa: E402

CACHE = reg.OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def text(t):
    return {"type": "text", "text": t}


def image(im):
    return {"type": "image_url", "image_url": {"url": m.data_url(im)}}


def video(path, fps=1):
    """Chat Completions video part. Base64 data URL, so no upload step."""
    return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(Path(path).read_bytes()).decode(), "fps": fps}}


def chat(content, schema=None, effort=None, model="muse-spark-1.3", max_tokens=24000, cache_key=None, kind="samtricks"):
    """Spark with the parts in the order given. spark() in metalab always puts text first and has no video part."""
    m._guard()
    kw = {"max_completion_tokens": max_tokens}
    if schema:
        kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}}
    if effort:
        kw["reasoning_effort"] = effort
    if cache_key:
        kw["prompt_cache_key"] = cache_key
    t0 = time.time()
    r = m._retry(lambda: m.client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], **kw))
    u = r.usage
    cached = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
    pi, pc, po = m.PRICES[model]
    usd = ((u.prompt_tokens - cached) * pi + cached * pc + u.completion_tokens * po) / 1e6
    m._log(model, usd, input=u.prompt_tokens, output=u.completion_tokens, cached=cached, kind=kind)
    t = r.choices[0].message.content
    return (json.loads(t) if schema else t), dict(usd=usd, seconds=round(time.time() - t0, 1), input=u.prompt_tokens, output=u.completion_tokens, cached=cached)


def sam(word, im, tag=""):
    """Concept call cached on word and image bytes. Returns rows of box [x1, y1, x2, y2], conf and mask area in px. A cached call costs nothing."""
    key = hashlib.sha1(word.encode() + cv2.imencode(".png", im)[1].tobytes()).hexdigest()
    p = CACHE / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    recs = m.sam_image(word, im)
    rows = [dict(id=int(r.object_id), box=[r.bounds.left, r.bounds.top, r.bounds.right, r.bounds.bottom], conf=None if r.confidence is None else round(float(r.confidence), 3), area=int(m.raster(r).sum())) for r in recs]
    p.write_text(json.dumps(rows))
    return rows


def pmap(fn, items, n=6):
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(fn, items))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def covered(truth, b):
    """Share of the truth box that falls inside b."""
    ix = max(0, min(truth[2], b[2]) - max(truth[0], b[0]))
    iy = max(0, min(truth[3], b[3]) - max(truth[1], b[1]))
    a = (truth[2] - truth[0]) * (truth[3] - truth[1])
    return ix * iy / a if a > 0 else 0.0


def match(preds, truths, thr):
    """One-to-one greedy match. Returns (pred index, truth index, iou) rows."""
    pairs = sorted(((iou(p, t), i, j) for i, p in enumerate(preds) for j, t in enumerate(truths)), reverse=True)
    used_p, used_t, out = set(), set(), []
    for v, i, j in pairs:
        if v >= thr and i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            out.append((i, j, v))
    return out


def to_grid(box, w, h):
    return [box[0] / w * 1000, box[1] / h * 1000, box[2] / w * 1000, box[3] / h * 1000]


def to_px(b, w, h):
    return [int(b[0] / 1000 * w), int(b[1] / 1000 * h), int(b[2] / 1000 * w), int(b[3] / 1000 * h)]


def largest_ink_box(im):
    """No-model baseline: bounding box of the largest ink component after closing gaps. Border rules are dropped by ignoring components that span the sheet."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ink = (g < 160).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((13, 13), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    h, w = g.shape
    best = None
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if bw > .96 * w or bh > .96 * h:
            continue
        if best is None or a > best[4]:
            best = (x, y, bw, bh, a)
    return to_grid([best[0], best[1], best[0] + best[2], best[1] + best[3]], w, h) if best else [0, 0, 780, 1000]


if __name__ == "__main__":
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9
    assert match([[0, 0, 10, 10], [0, 0, 9, 9]], [[0, 0, 10, 10]], .5) == [(0, 0, 1.0)]
    assert covered([0, 0, 10, 10], [0, 0, 5, 10]) == .5
    print("ok")
