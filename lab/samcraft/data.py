"""Test set: CubiCasa5K plans read from the zip, with room, stair and door polygons in image pixels.
Evaluation only (CC BY-NC-SA): nothing here goes in the demo or in training. SVG polygons line up with F1_scaled.png."""
import json
import random
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/lab/samcraft"
ZIP = ROOT / "data/plans/cubicasa5k/cubicasa5k.zip"
SKIP = ("Outdoor", "Undefined", "Background", "CarPort", "Elevated")
zf = zipfile.ZipFile(ZIP)


def _pts(txt):
    n = np.array(re.findall(r"-?\d+\.?\d*", txt), float)
    return n[: len(n) // 2 * 2].reshape(-1, 2)


def _groups(s, cls_re):
    for m in re.finditer(r'<g[^>]*class="(%s)"[^>]*>\s*<polygon points="([^"]+)"' % cls_re, s):
        yield m.group(1), _pts(m.group(2))


def load(pid):
    """Floor 1 of a plan: BGR image and polygons by kind."""
    base = f"cubicasa5k/high_quality_architectural/{pid}"
    s = zf.read(f"{base}/model.svg").decode()
    im = cv2.imdecode(np.frombuffer(zf.read(f"{base}/F1_scaled.png"), np.uint8), cv2.IMREAD_COLOR)
    if 'id="Floor-1" transform' in s and 'id="Floor-1" transform="matrix(1,0,0,1,0,0)"' not in s:
        raise ValueError("floor transform")
    s = s.split('class="Floorplan Floor-2"')[0]
    rooms = [(c.replace("Space ", ""), q) for c, q in _groups(s, r"Space [^\"]*") if not any(k in c for k in SKIP)]
    return im, {"rooms": rooms, "stairs": list(_groups(s, r"Flight|Winding|RoundedWinding")), "doors": list(_groups(s, r"Door [^\"]*"))}


def ink_of(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return g < 160


def near_ink(q, d, tol=6):
    """Share of the polygon outline with ink within tol px, given d, the distance map to ink: is a wall really drawn there."""
    pts = []
    for a, b in zip(q, np.roll(q, -1, 0)):
        n = max(int(np.linalg.norm(b - a) / 3), 1)
        pts += [a + (b - a) * t / n for t in range(n)]
    pts = np.array(pts).round().astype(int)
    pts[:, 0] = pts[:, 0].clip(0, d.shape[1] - 1)
    pts[:, 1] = pts[:, 1].clip(0, d.shape[0] - 1)
    return float((d[pts[:, 1], pts[:, 0]] <= tol).mean())


def poly_mask(q, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    return m.astype(bool)


def rep_point(q):
    p = Polygon(q).buffer(0).representative_point()
    return float(p.x), float(p.y)


def select():
    """A fixed sample of plans. Cached so the room list never changes between runs."""
    path = OUT / "testset.json"
    if path.exists():
        return json.loads(path.read_text())
    ids = sorted({x.split("/")[2] for x in zf.namelist() if x.endswith("model.svg") and "high_quality_architectural" in x}, key=int)
    random.Random(0).shuffle(ids)
    plans, stairs, rooms = [], [], []
    for pid in ids[:1200]:  # shortcut: stops at 1200 plans so a stair-poor shuffle cannot scan the whole zip
        if len(plans) >= 12:
            break
        try:
            im, g = load(pid)
        except (ValueError, KeyError, AttributeError):
            continue
        if not 900 <= max(im.shape[:2]) <= 2200 or len(g["rooms"]) < 4:
            continue
        d = ndi.distance_transform_edt(~ink_of(im))
        good = [i for i, (c, q) in enumerate(g["rooms"]) if len(q) >= 3 and Polygon(q).buffer(0).area >= 2500]
        scores = {i: near_ink(g["rooms"][i][1], d) for i in good}
        if not good or np.median(list(scores.values())) < 0.6:  # polygons that do not sit on the drawing
            continue
        if len(plans) < 12:
            plans.append(pid)
            pick = random.Random(int(pid)).sample(good, min(6, len(good)))
            rooms += [dict(pid=pid, i=i, cls=g["rooms"][i][0], walled=scores[i] >= 0.85) for i in pick]
    for pid in ids[:1200]:
        if len(stairs) >= 12:
            break
        if 'class="Flight"' in zf.read(f"cubicasa5k/high_quality_architectural/{pid}/model.svg").decode():
            try:
                im, g = load(pid)
            except (ValueError, KeyError, AttributeError):
                continue
            if 900 <= max(im.shape[:2]) <= 2200 and g["stairs"]:
                stairs.append(pid)
    res = dict(plans=plans, stairs=stairs, rooms=rooms)
    path.write_text(json.dumps(res, indent=0))
    return res


if __name__ == "__main__":
    t = select()
    print(len(t["plans"]), "plans", len(t["rooms"]), "rooms", sum(r["walled"] for r in t["rooms"]), "walled", len(t["stairs"]), "stair plans")
    tiles = []
    for pid in t["plans"][:3] + t["stairs"][:3]:
        im, g = load(pid)
        v = im.copy()
        for c, q in g["rooms"]:
            cv2.polylines(v, [q.astype(np.int32)], True, (0, 0, 255), 2)
        for c, q in g["doors"]:
            cv2.polylines(v, [q.astype(np.int32)], True, (255, 0, 0), 3)
        for c, q in g["stairs"]:
            cv2.polylines(v, [q.astype(np.int32)], True, (255, 0, 255), 3)
        tiles.append(cv2.resize(v, (640, 640), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(OUT / "gt_check.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
