"""Planted defects with lab/defects/plant.py erase, on the raw sheet at the truth box, inset 2 px as lab/defects/spark_only.py does. No API here.
Dev: one stack per building with all three cases on different stairs, used only to tune. Held-out: one stack per case. Registration is reused: an erased stair is far too small to move a fit with hundreds of inliers."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/presence")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/defects")
import pres
from pres import OUT, SH, T, TRUTH
from plant import erase

P = OUT / "planted"


def build(b, key, cases):
    d = P / key
    d.mkdir(parents=True, exist_ok=True)
    floors = {}
    for kind, o, f in cases:
        floors.setdefault(f, []).append((kind, o))
    tiles = []
    for f, items in floors.items():
        im = cv2.imread(str(SH / f"{b}/{f}.jpg"))
        before = im.copy()
        for k, (kind, o) in enumerate(items):
            x1, y1, x2, y2 = T[b]["objects"][o]["boxes"][str(f)]
            im = erase(im, x1 + 2, y1 + 2, x2 - 2, y2 - 2, seed=k + 1)
        cv2.imwrite(str(d / f"{f}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for kind, o in items:
            x1, y1, x2, y2 = T[b]["objects"][o]["boxes"][str(f)]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = max(x2 - x1, y2 - y1) // 2 + 50
            cut = lambda a: cv2.resize(cv2.copyMakeBorder(a, r, r, r, r, cv2.BORDER_CONSTANT, value=(180, 180, 180))[cy:cy + 2 * r, cx:cx + 2 * r], (260, 260), interpolation=cv2.INTER_CUBIC)
            t = np.hstack([cut(before), cut(im)])
            cv2.putText(t, f"{key} {kind} {o} f{f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (200, 0, 0), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    return tiles


def paths(b, key, order):
    return [(P / key / f"{i}.jpg") if (P / key / f"{i}.jpg").exists() else SH / f"{b}/{i}.jpg" for i in order]


DEV_KEYS = {f"{b}__all": (b, TRUTH["plant"][b]) for b in ("bme_f", "habs_stelizabeths_center")}
HELD_KEYS = {f"{b}__{c[0]}": (b, [c]) for b in pres.HELD for c in TRUTH["plant"][b]}
if __name__ == "__main__":
    which = DEV_KEYS if sys.argv[1] == "dev" else HELD_KEYS
    tiles = []
    for key, (b, cases) in which.items():
        tiles += build(b, key, cases)
    n = 3
    rows = [np.hstack(tiles[i:i + n] + [np.zeros_like(tiles[0])] * (n - len(tiles[i:i + n]))) for i in range(0, len(tiles), n)]
    cv2.imwrite(str(OUT / f"planted_{sys.argv[1]}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("planted", len(tiles))
