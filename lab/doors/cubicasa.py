"""Residential door check against CubiCasa5K annotations, read from the zip. Evaluation only: no images go in the demo."""
import json, random, re, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "lab"); sys.path.insert(0, str(Path(__file__).parent))
import metalab as m
import key as K
from spark_tiles import PROMPT, SCHEMA, CACHE, nms, score, tiles

Z = zipfile.ZipFile(K.ROOT / "data/plans/cubicasa5k/cubicasa5k.zip")
names = sorted(x for x in Z.namelist() if "high_quality/" in x and x.endswith("model.svg"))
random.seed(7)
random.shuffle(names)
picked = []
for n in names:
    d = n.rsplit("/", 1)[0]
    if f"{d}/F2_scaled.png" not in Z.namelist() and f"{d}/F1_scaled.png" in Z.namelist():
        svg = Z.read(n).decode(errors="replace")
        doors = re.findall(r'<g id="Door"[^>]*>\s*<polygon points="([^"]+)"', svg)
        if 4 <= len(doors) <= 30:
            picked.append((d, doors))
    if len(picked) == 8:
        break


def centre(s):
    p = np.array([[float(v) for v in t.split(",")] for t in s.split()])
    return p.mean(0).tolist(), float(np.linalg.norm(p.max(0) - p.min(0)))


def call(pid, name, im, rect, effort):
    f = CACHE / f"cc_{pid}_{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    x0, y0, x1, y1 = [int(v) for v in rect]
    crop = im[y0:y1, x0:x1]
    r, info = m.spark(PROMPT, images=[crop], schema=SCHEMA, effort=effort)
    o = {"rect": [x0, y0, x1, y1], "pts": r["points"], "usd": info["usd"]}
    f.write_text(json.dumps(o))
    return o


def pts(o):
    x0, y0, x1, y1 = o["rect"]
    return [[x0 + p["x"] / 1000 * (x1 - x0), y0 + p["y"] / 1000 * (y1 - y0)] for p in o["pts"]]


rows = {"whole": [0, 0, 0, 0.0], "t2": [0, 0, 0, 0.0]}
for d, doors in picked:
    pid = d.rsplit("/", 1)[1]
    im = cv2.imdecode(np.frombuffer(Z.read(f"{d}/F1_scaled.png"), np.uint8), 1)
    G = [centre(s)[0] for s in doors]
    w = np.median([centre(s)[1] for s in doors])
    H, W = im.shape[:2]
    a = call(pid, "whole", im, (0, 0, W, H), "low")
    with ThreadPoolExecutor(4) as ex:
        ts = list(ex.map(lambda it: call(pid, f"t2_{it[0]}", im, it[1], "low"), enumerate(tiles((0, 0, W, H), 2))))
    for key, P, usd in (("whole", pts(a), a["usd"]), ("t2", nms([p for o in ts for p in pts(o)], 0.6 * w), sum(o["usd"] for o in ts))):
        tp = score(P, G, 1.0 * w)
        rows[key][0] += tp; rows[key][1] += len(P); rows[key][2] += len(G); rows[key][3] += usd
    print(pid, W, H, len(G), rows, flush=True)
res = {k: {"tp": v[0], "pred": v[1], "truth": v[2], "recall": v[0] / v[2], "precision": v[0] / max(v[1], 1), "usd": v[3], "plans": len(picked)} for k, v in rows.items()}
print(res)
(K.OUT / "cubicasa.json").write_text(json.dumps(res, indent=1))
