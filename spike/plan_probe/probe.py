"""Throwaway probe, not product code: can Spark read a real floor plan, can SAM mask it, and can SAM's video tracker
follow a staircase through a stack of floors? Run from the plans folder with MODEL_API_KEY set. Spend under 0.10 USD."""
import base64
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format, video_segmentation_format
from openai import OpenAI

client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], timeout=300)
W = 2000


def load(path):
    im = cv2.imread(path)
    return cv2.resize(im, (W, round(im.shape[0] * W / im.shape[1])), interpolation=cv2.INTER_AREA)


def b64(im):
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]).decode()


def sam_text(prompt, media):
    stream = client.responses.create(
        model="sam-3.1", stream=True, metadata={"include_confidence": "true", "propagation_direction": "both"} if media["type"] == "input_video" else {"include_confidence": "true"},
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}, media]}])
    text = ""
    for ev in stream:
        ev = ev.model_dump()
        if ev.get("type") == "response.output_text.delta":
            text += ev["delta"]
        elif ev.get("type") in ("response.failed", "response.incomplete", "error"):
            raise RuntimeError(json.dumps(ev, default=str)[:400])
    return text


def records(text, fmt):
    parser = fmt().create_parser()
    parser.push(text if text.endswith("\n") else text + "\n")
    return [r for r in parser.finish(CompletedOutcome()).result.records if r.kind == "mask"]


def raster(rec):
    m = rec.mask
    return np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)


def paint(canvas, rec, color):
    b = rec.bounds
    r = cv2.resize(raster(rec), (b.right - b.left, b.bottom - b.top), interpolation=cv2.INTER_NEAREST).astype(bool)
    h, w = canvas.shape[:2]
    y2, x2 = min(b.bottom, h), min(b.right, w)
    roi = canvas[b.top:y2, b.left:x2]
    roi[r[: y2 - b.top, : x2 - b.left]] = (0.45 * roi[r[: y2 - b.top, : x2 - b.left]] + 0.55 * np.array(color)).astype(np.uint8)


def spark(im):
    schema = {"type": "object", "additionalProperties": False, "required": ["title", "floor", "stairs", "lecture_halls", "rooms_seen"], "properties": {
        "title": {"type": "string"}, "floor": {"type": "string"}, "rooms_seen": {"type": "integer"},
        "stairs": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["label", "x", "y"], "properties": {"label": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}}}},
        "lecture_halls": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "area_m2", "seats", "x", "y"], "properties": {"name": {"type": "string"}, "area_m2": {"type": "number"}, "seats": {"type": "integer"}, "x": {"type": "integer"}, "y": {"type": "integer"}}}},
    }}
    prompt = ("This is a scanned architectural floor plan with Hungarian labels. Report the sheet title and the floor. List every staircase with its "
              "label and the centre of its flight of steps. List every lecture hall (ELOADOTEREM) with the area in m2 and seat count (FO) written on it. "
              "Count the labelled rooms. Coordinates are integers on a 0-1000 grid, x from the left edge and y from the top edge of the image.")
    r = client.chat.completions.create(model="muse-spark-1.3", response_format={"type": "json_schema", "json_schema": {"name": "plan", "schema": schema, "strict": True}},
                                       messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(im)}}]}])
    return json.loads(r.choices[0].message.content), r.usage


def align(mov, ref):
    """Similarity transform from sheet `mov` onto sheet `ref`, using features of the shared building outline."""
    orb = cv2.ORB_create(8000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    src, dst = np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kb[x.trainIdx].pt for x in m])
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6, maxIters=20000)
    return cv2.warpAffine(mov, A, (ref.shape[1], ref.shape[0]), borderValue=(220, 230, 235)), int(inl.sum())


def stack(frames, path, repeat=4):
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}", "-r", "4", "-i", "-",
                          "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in frames:
        for _ in range(repeat):
            p.stdin.write(f[:h2, :w2].tobytes())
    p.stdin.close()
    p.wait()
    return w2, h2


def track(path, boxes, w, h, floors, repeat=4):
    """Prompt on the first frame of the ground floor and report, for each staircase, its mask on each floor."""
    with open(path, "rb") as f:
        fid = client.files.create(purpose="user_data", file=f).id
    start = floors.index("level 0") * repeat
    prompt = f"<{start}f>" + ",".join(f"{i}<|box;x1={b[0]};y1={b[1]};x2={b[2]};y2={b[3]};w={w};h={h}|>" for i, b in boxes.items())
    recs = records(sam_text(prompt, {"type": "input_video", "file_id": fid}), video_segmentation_format)
    out = {}
    for r in recs:
        floor = floors[r.frame.frame_index // repeat]
        a = int(raster(r).sum()) if r.frame.frame_index % repeat == repeat - 1 else None
        if a is not None:
            b = r.bounds
            out.setdefault(r.object_id, {})[floor] = dict(area=a, box=[b.left, b.top, b.right, b.bottom], conf=None if r.confidence is None else round(r.confidence, 3))
    return out, recs


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "spark":
        for f in ("bme_1.jpg", "bme_0.jpg"):
            d, u = spark(load(f))
            print(f, json.dumps(d, ensure_ascii=False))
            print(u)
    if step == "concept":
        im = load("bme_1.jpg")
        data = {"type": "input_image", "image_url": "data:image/jpeg;base64," + b64(im)}
        for c in ("staircase", "stairs", "door", "room", "hallway", "wall"):
            rs = records(sam_text(c, data), image_segmentation_format)
            over = im.copy()
            for r in rs[:60]:
                paint(over, r, (255, 0, 200))
            cv2.imwrite(f"sam_{c}.jpg", over)
            print(c, len(rs), [(r.bounds.left, r.bounds.top, r.bounds.right, r.bounds.bottom, r.confidence and round(r.confidence, 2)) for r in rs[:6]])
    if step == "room":
        im = load("bme_1.jpg")
        data = {"type": "input_image", "image_url": "data:image/jpeg;base64," + b64(im)}
        h, w = im.shape[:2]
        pts = {1: (1330, 430), 2: (475, 430), 3: (875, 1125), 4: (900, 930), 5: (350, 790)}   # two lecture halls, classroom, stair I, lab
        prompt = "<0f>" + ",".join(f"{i}<|point;x={x};y={y};w={w};h={h}|>" for i, (x, y) in pts.items())
        rs = records(sam_text(prompt, data), image_segmentation_format)
        over = im.copy()
        colors = {1: (255, 0, 200), 2: (0, 160, 255), 3: (0, 200, 0), 4: (255, 120, 0), 5: (0, 0, 255)}
        for r in rs:
            paint(over, r, colors.get(r.object_id, (0, 0, 0)))
            print(r.object_id, (r.bounds.left, r.bounds.top, r.bounds.right, r.bounds.bottom), int(raster(r).sum()), r.confidence)
        cv2.imwrite("sam_rooms.jpg", over)

    if step == "stairs":
        names = ["level 2", "level 1", "level 0", "level -1"]
        ref = load("bme_0.jpg")
        frames = []
        for f in ("bme_2.jpg", "bme_1.jpg", "bme_0.jpg", "bme_-1.jpg"):
            al, inl = (ref, -1) if f == "bme_0.jpg" else align(load(f), ref)
            print(f, "inliers", inl)
            frames.append(al)
        # staircases on the ground floor, boxes read by eye from the 2000 px sheet: I, II, III, top left, top right
        boxes = {1: (848, 900, 945, 1015), 2: (1175, 660, 1275, 740), 3: (512, 652, 612, 735), 4: (418, 215, 530, 280), 5: (1275, 228, 1385, 290)}
        w, h = stack(frames, "floors.mp4")
        res, recs = track("floors.mp4", boxes, w, h, names)
        print("REAL STACK")
        for k in sorted(res):
            print(k, res[k])
        # negative test: erase staircase II from level 1 by painting paper colour, keep everything else
        erased = [f.copy() for f in frames]
        x1, y1, x2, y2 = boxes[2]
        erased[1][y1 - 10:y2 + 10, x1 - 10:x2 + 10] = np.median(frames[1].reshape(-1, 3), axis=0)
        w, h = stack(erased, "floors_erased.mp4")
        res2, _ = track("floors_erased.mp4", {2: boxes[2], 1: boxes[1]}, w, h, names)
        print("STAIR II ERASED ON LEVEL 1")
        for k in sorted(res2):
            print(k, res2[k])
        strip = [cv2.resize(f, (960, round(f.shape[0] * 960 / f.shape[1]))) for f in frames]
        cv2.imwrite("floors_strip.jpg", np.vstack([np.hstack(strip[:2]), np.hstack(strip[2:])]))
