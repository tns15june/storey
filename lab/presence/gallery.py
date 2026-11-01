"""Read-by-eye sheets of decided cells. No API. Usage: gallery.py heldout. Writes data/lab/presence/gal_<name>.jpg: phantoms, false findings, Found, Check."""
import json
import sys
from collections import Counter

sys.path.insert(0, "lab/presence")
import cv2
import numpy as np

import planted
import pres
import rule
import score as S
from pres import OUT, SH, T

split = sys.argv[1]
B = [b for b, t in T.items() if t["split"] == split]
ARM = "C"


def path(key, b, f):
    return planted.P / key / f"{f}.jpg" if (planted.P / key / f"{f}.jpg").exists() else SH / f"{b}/{f}.jpg"


def tile(key, b, f, loc, text, half=95, zoom=1.6):
    im = cv2.imread(str(path(key, b, f)))
    x, y = int(loc[0]), int(loc[1])
    P = cv2.copyMakeBorder(im, half, half, half, half, cv2.BORDER_CONSTANT, value=(180, 180, 180))
    t = cv2.resize(P[y:y + 2 * half, x:x + 2 * half], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    c = int(half * zoom)
    cv2.drawMarker(t, (c, c), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
    cv2.putText(t, text[:42], (2, 11), cv2.FONT_HERSHEY_SIMPLEX, .36, (200, 0, 0), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def sheet(name, tiles, n=7):
    if not tiles:
        return
    rows = [np.hstack(tiles[i:i + n] + [np.zeros_like(tiles[0])] * (n - len(tiles[i:i + n]))) for i in range(0, len(tiles), n)]
    for k in range(0, len(rows), 6):
        cv2.imwrite(str(OUT / f"gal_{name}{k // 6}.jpg"), np.vstack(rows[k:k + 6]), [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(name, len(tiles))


keys = [(b, b) for b in B] + [(k, v[0]) for k, v in (planted.DEV_KEYS if split == "dev" else planted.HELD_KEYS).items() if (OUT / f"cells/{k}.json").exists()]
ph, fa = [], []
for key, b in keys:
    E = json.load(open(OUT / f"cells/{key}.json"))
    for F in ("stair", "shaft"):
        d = [c for c in rule.apply_rule(E, ARM) if c["fam"] == F]
        s = S.score(b, d, F)
        for x in s["wrong"] + s["phantom"]:
            r = E["rows"][x["row"]]
            c = r["cells"][str(x["f"])]
            ph.append((key, b, x["f"], x["loc"], f"{key[:20]} f{x['f']} v{c['v']} {c.get('crop')} {r['ink30'].get(str(x['f']))}", (b, x["f"], round(x["loc"][0] / 40) * 40, round(x["loc"][1] / 40) * 40)))
        for x in s["false"]:
            r = E["rows"][x["row"]]
            c = r["cells"][str(x["f"])]
            fa.append((key, b, x["f"], x["loc"], f"{key[:18]} f{x['f']} {x['obj']} v{c['v']} {c.get('crop')}", (b, x["f"], x["obj"])))
print("phantom places", Counter(p[5] for p in ph).most_common())
print("false places", Counter(p[5] for p in fa).most_common())
seen, t = set(), []
for p in ph:
    if p[5] not in seen:
        seen.add(p[5])
        t.append(tile(*p[:5]))
sheet("phantom", t)
seen, t = set(), []
for p in fa:
    if p[5] not in seen:
        seen.add(p[5])
        t.append(tile(*p[:5]))
sheet("false", t)
for st in ("Found", "Check", "Missing"):
    t = []
    for b in B:
        E = json.load(open(OUT / f"cells/{b}.json"))
        for c in rule.apply_rule(E, ARM):
            if c["state"] == st:
                t.append(tile(b, b, c["f"], c["loc"] or c["chk"], f"{b[:10]} {c['fam'][:2]} f{c['f']} v{c['v']} {c['why'][:22]}", half=70, zoom=1.2))
    sheet(st, t, 9)
