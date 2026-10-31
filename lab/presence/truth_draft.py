"""No API. Carry earlier stair boxes onto this experiment's raw 2000 px sheets, as a draft to check by eye. Writes draft.json and one gallery per building."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/Users/tsharma/Github/Meta")
sys.path.insert(0, str(ROOT / "lab/defects"))
sys.path.insert(0, str(ROOT / "lab/align"))
from seeds import SEEDS, TRUTH

OUT = ROOT / "data/lab/presence"
SH = OUT / "sheets"
V = OUT / "truth_views"
INDEX = json.load(open(SH / "index.json"))
AUD = json.load(open(ROOT / "lab/audit/truth.json"))


def sheet(b, i):
    return cv2.imread(str(SH / b / f"{i}.jpg"))


def fit(src, dst):
    """Similarity from one rendering of a page to another, by ORB. Same source image, so it needs no gate beyond inliers."""
    orb = cv2.ORB_create(20000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    a = np.float32([ka[x.queryIdx].pt for x in m]); b = np.float32([kb[x.trainIdx].pt for x in m])
    A, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=4, maxIters=20000)
    return np.vstack([A, [0, 0, 1]]), int(inl.sum())


def map_box(T, b):
    p = np.array([[b[0], b[1], 1], [b[2], b[3], 1]], float) @ T.T
    return [int(round(v)) for v in (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())]


draft = {}
# earlier defect stacks: SEEDS boxes on stack images
for b in ("bme_f", "habs_stelizabeths_center", "schenley", "mml_capitol"):
    d = {}
    lv = [str(r["level"]) for r in INDEX[b]]
    T = {}
    for i, l in enumerate(lv):
        st = cv2.imread(str(ROOT / f"data/lab/defects/stack/{b}/{l}.png"))
        T[i], n = fit(st, sheet(b, i))
        print(b, l, "fit inliers", n, "scale", round(float(np.hypot(*T[i][:2, 0])), 3), flush=True)
    for o, (kind, box) in SEEDS[b]["objects"].items():
        d[o] = {"kind": kind, "boxes": {str(i): map_box(T[i], box) for i, l in enumerate(lv)}, "prior": [str(lv.index(x)) for x in TRUTH[b][o]]}
    draft[b] = d
# audit truth: per-floor boxes on stacks made with data/lab/audit/<b>/align.json
for b in AUD:
    if b.startswith("_"):
        continue
    al = json.load(open(ROOT / f"data/lab/audit/{b}/align.json"))
    n = len(INDEX[b])
    s = sheet(b, 0).shape[1] / 2000  # audit rendered 2000 px wide, this experiment 2000 px on the long side
    d = {}
    Tinv = {}
    for i in range(n):
        A = al[str(i)].get("A")
        M = np.eye(3) if A is None else np.vstack([np.array(A, float), [0, 0, 1]])
        Tinv[i] = np.diag([s, s, 1.0]) @ np.linalg.inv(M)
    for o, v in AUD[b].items():
        any_box = next(iter(v["boxes"].values()))
        d[o] = {"kind": "stair", "cls": v["cls"], "boxes": {str(i): map_box(Tinv[i], v["boxes"].get(str(i), any_box)) for i in range(n)}, "prior": list(v["boxes"])}
    draft[b] = d
json.dump(draft, open(OUT / "draft.json", "w"), indent=1)


def gallery(b, d, half=110, zoom=1.5):
    n = len(INDEX[b])
    ims = [sheet(b, i) for i in range(n)]
    S = int(2 * half * zoom)
    rows = []
    for o, v in d.items():
        ts = []
        for i in range(n):
            x1, y1, x2, y2 = v["boxes"][str(i)]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pad = cv2.copyMakeBorder(ims[i], half, half, half, half, cv2.BORDER_CONSTANT, value=(180, 180, 180))
            t = cv2.resize(pad[cy:cy + 2 * half, cx:cx + 2 * half], (S, S), interpolation=cv2.INTER_CUBIC)
            q = lambda a, c: int((a - c + half) * zoom)
            on = str(i) in v["prior"]
            cv2.rectangle(t, (q(x1, cx), q(y1, cy)), (q(x2, cx), q(y2, cy)), (0, 160, 0) if on else (0, 0, 255), 1)
            cv2.putText(t, f"{o} f{i} {'P' if on else 'a'} {cx},{cy}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, .42, (200, 0, 0), 1, cv2.LINE_AA)
            ts.append(cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        rows.append(np.hstack(ts))
    for k in range(0, len(rows), 5):
        cv2.imwrite(str(V / f"{b}_g{k // 5}.jpg"), np.vstack(rows[k:k + 5]), [cv2.IMWRITE_JPEG_QUALITY, 88])


for b, d in draft.items():
    gallery(b, d)
print("done")
