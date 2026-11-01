"""Overlay of the frozen item 2 truth, no model. Run before any word call to catch labelling slips."""
import json
import sys

sys.path.insert(0, "lab/samtricks")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import reg  # noqa: E402

T = json.load(open(reg.TRUTH / "item2_words.json"))["truth"]
C = json.load(open(reg.TRUTH / "item4_clouds.json"))["clouds"]
COL = {"title block": (0, 0, 255), "table": (0, 160, 0), "schedule": (255, 120, 0), "north arrow": (255, 0, 255), "scale bar": (0, 200, 200), "stamp": (200, 0, 200), "legend": (120, 60, 0)}
cells = []
for k, v in T.items():
    im = reg.sheet(k).copy()
    h, w = im.shape[:2]
    P = lambda b: ((int(b[0] / 1000 * w), int(b[1] / 1000 * h)), (int(b[2] / 1000 * w), int(b[3] / 1000 * h)))
    for word, t in v.items():
        if t == "clouds":
            continue
        for b in t["req"]:
            cv2.rectangle(im, *P(b), COL[word], 3)
            cv2.putText(im, word, (P(b)[0][0], max(P(b)[0][1] - 4, 14)), 0, .6, COL[word], 2)
        for b in t.get("opt", []):
            cv2.rectangle(im, *P(b), COL[word], 1)
    c = cv2.resize(im, (1000, 700), interpolation=cv2.INTER_AREA)
    cv2.putText(c, k, (6, 26), 0, .9, (0, 0, 255), 2)
    cells.append(c)
for i in range(0, 12, 4):
    cv2.imwrite(str(reg.OUT / f"preview/words_truth{i // 4}.jpg"), np.vstack([np.hstack(cells[i:i + 2]), np.hstack(cells[i + 2:i + 4])]), [cv2.IMWRITE_JPEG_QUALITY, 82])
