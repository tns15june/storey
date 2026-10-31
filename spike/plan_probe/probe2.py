"""Throwaway: (a) Spark alone on the two sheets not yet read, as the no-SAM baseline; (b) Spark judging crops at SAM's tracked boxes."""
import json, subprocess, sys
import cv2, numpy as np
sys.argv = ["probe.py", "none"]
from probe import spark, load, client, b64

for f in ("bme_2.jpg", "bme_-1.jpg"):
    d, _ = spark(load(f))
    print(f, d["floor"], [(s["label"], s["x"], s["y"]) for s in d["stairs"]])

def frame(path, n):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{n})", "-fps_mode", "passthrough", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

def ask(im, box, pad=40):
    x1, y1, x2, y2 = box
    crop = im[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    schema = {"type": "object", "additionalProperties": False, "required": ["stair_steps_visible", "what_is_here"], "properties": {"stair_steps_visible": {"type": "boolean"}, "what_is_here": {"type": "string"}}}
    r = client.chat.completions.create(model="muse-spark-1.3", response_format={"type": "json_schema", "json_schema": {"name": "crop", "schema": schema, "strict": True}},
        messages=[{"role": "user", "content": [{"type": "text", "text": "This is a crop of a scanned floor plan. Does it show a staircase, meaning a run of parallel step lines? Answer from what is drawn."}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(crop)}}]}])
    return json.loads(r.choices[0].message.content)

# frame index = floor * 4 + 3; floors: 0 level 2, 1 level 1, 2 level 0, 3 level -1
cases = [
 ("erased stair II, level 1 (truth: none)", "floors_erased.mp4", 1, (1177, 750, 1276, 776)),
 ("real small stair 4, level 1 (truth: stair)", "floors.mp4", 1, (442, 259, 527, 290)),
 ("real stair 5, level 1 (truth: stair)", "floors.mp4", 1, (1260, 266, 1368, 304)),
 ("real stair 4, level -1 (truth: stair)", "floors.mp4", 3, (428, 273, 534, 314)),
 ("real stair II, level 1 (truth: stair)", "floors.mp4", 1, (1175, 690, 1280, 774)),
]
for name, path, fl, box in cases:
    print(name, "->", ask(frame(path, fl * 4 + 3), box))
