"""No API. One tile per object and floor where a box was newly read, to see that each box sits on its object. Reads truth_src objects."""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path("/Users/tsharma/Github/Meta"); OUT = ROOT / "data/lab/presence"
T = json.load(open(ROOT / "lab/presence/truth.json"))["buildings"]; D = json.load(open(OUT / "draft.json"))
tiles = []
for b, t in T.items():
    for o, v in t["objects"].items():
        if o in D.get(b, {}) and v["boxes"] and all(D[b][o]["boxes"].get(f) == x for f, x in v["boxes"].items()):
            continue
        for f, bx in sorted(v["boxes"].items()):
            im = cv2.imread(str(OUT / f"sheets/{b}/{f}.jpg")); pad = 40
            x1, y1, x2, y2 = bx
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2; r = max(x2 - x1, y2 - y1) // 2 + pad
            P = cv2.copyMakeBorder(im, r, r, r, r, cv2.BORDER_CONSTANT, value=(180, 180, 180))
            c = P[cy:cy + 2 * r, cx:cx + 2 * r].copy()
            z = 150 / c.shape[0]
            c = cv2.resize(c, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(c, (int((x1 - cx + r) * z), int((y1 - cy + r) * z)), (int((x2 - cx + r) * z), int((y2 - cy + r) * z)), (0, 170, 0), 1)
            cv2.putText(c, f"{b[:6]} {o[:9]} f{f}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, .32, (200, 0, 0), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(c, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
n = 10
rows = [np.hstack(tiles[i:i + n] + [np.zeros_like(tiles[0])] * (n - len(tiles[i:i + n]))) for i in range(0, len(tiles), n)]
for k in range(0, len(rows), 7):
    cv2.imwrite(str(OUT / f"truth_views/check{k // 7}.jpg"), np.vstack(rows[k:k + 7]), [cv2.IMWRITE_JPEG_QUALITY, 90])
print(len(tiles))
