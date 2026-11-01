"""Second try at clinic exits: the exterior door prompt on tiles, and reading the architect's exit arrows on the life safety sheets."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, fitz, numpy as np
sys.path.insert(0, "lab"); sys.path.insert(0, str(Path(__file__).parent))
import metalab as m
import key as K
from exits import PROMPT
from spark_tiles import SCHEMA, CACHE, tiles, nms, score, to_sheet

SCH = {"type": "object", "additionalProperties": False, "required": ["exits"], "properties": {"exits": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["x", "y", "text"], "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "text": {"type": "string"}}}}}}
LS = ('This is part of a life safety plan. Point to every exit marker: a short arrow with an exit width and a capacity in people written beside it, such as "915 / 180 People", '
      'or a box reading "STAIR WELL EXIT CAPACITY". Give the x and y of the arrow or box on a 0-1000 grid and copy the text exactly into text. Ignore travel distance arrows, door swing arcs and other symbols.')
LS_SHEETS = {"F1": "Clinic_161_Life Safety Composite First Floor Plan", "F2": "Clinic_162_Life Safety Composite Second Floor Plan"}


def tile(tag, i, rect):
    f = CACHE / f"exittile_{tag}_{i}.json"
    if f.exists():
        return json.loads(f.read_text())
    dpi = int(round(72 * 1600 / max(rect[2] - rect[0], rect[3] - rect[1])))
    im = K.render(tag, dpi, clip=fitz.Rect(*rect))
    r, info = m.spark(PROMPT, images=[im], schema=SCHEMA, effort="low")
    out = {"rect": list(rect), "doors": r["points"], "info": info}
    f.write_text(json.dumps(out))
    return out


def lifesafety(tag, i, rect):
    f = CACHE / f"ls_{tag}_{i}.json"
    if f.exists():
        return json.loads(f.read_text())
    p = fitz.open(str(K.DIR / (LS_SHEETS[tag] + ".pdf")))[0]
    dpi = int(round(72 * 2000 / max(rect[2] - rect[0], rect[3] - rect[1])))
    pix = p.get_pixmap(dpi=dpi, clip=fitz.Rect(*rect))
    im = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3][:, :, ::-1].copy()
    r, info = m.spark(LS, images=[im], schema=SCH)
    out = {"rect": list(rect), "exits": r["exits"], "usd": info["usd"]}
    f.write_text(json.dumps(out))
    return out


res = {}
for tag in K.SHEETS:
    A, L, arcs, reg = K.clinic_key(tag)
    h = np.array([l["s_hinge"] for l in L])
    box = (h[:, 0].min() - 80, h[:, 1].min() - 80, h[:, 0].max() + 80, h[:, 1].max() + 80)
    ts = tiles(box, 3)
    with ThreadPoolExecutor(6) as ex:
        outs = list(ex.map(lambda it: tile(tag, it[0], it[1]), enumerate(ts)))
    pts = nms([p for o in outs for p in to_sheet(o)], 1.5 * K.PT_PER_M)
    T = {}
    for l in L:
        if l["external"] and "Chain Link" not in l["name"]:
            T.setdefault(l["door"], []).append([(l["s_hinge"][0] + l["s_shut"][0]) / 2, (l["s_hinge"][1] + l["s_shut"][1]) / 2])
    T = [np.mean(v, axis=0).tolist() for v in T.values()]
    res[tag] = {"tiles": {"found": len(pts), "truth": len(T), "tp": score(pts, T, 1.5 * K.PT_PER_M), "usd": sum(o["info"]["usd"] for o in outs)}}
    page = K.page(tag).rect
    with ThreadPoolExecutor(4) as ex:
        ls = list(ex.map(lambda it: lifesafety(tag, it[0], it[1]), enumerate(tiles((600, 100, 2150, 1500), 2))))
    seen = []
    for o in ls:
        x0, y0, x1, y1 = o["rect"]
        for e in o["exits"]:
            seen.append({"pt": [x0 + e["x"] / 1000 * (x1 - x0), y0 + e["y"] / 1000 * (y1 - y0)], "text": e["text"]})
    keep = []
    for e in seen:
        if all(np.hypot(e["pt"][0] - k["pt"][0], e["pt"][1] - k["pt"][1]) > 40 for k in keep):
            keep.append(e)
    res[tag]["life_safety_read"] = keep
    res[tag]["life_safety_usd"] = sum(o["usd"] for o in ls)
    print(tag, json.dumps(res[tag]), flush=True)
(K.OUT / "exits_tiles.json").write_text(json.dumps(res, indent=1))
