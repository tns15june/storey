"""Exact-truth test on real floor pairs: apply a known transform to the mov sheet, register, and measure the landmark error.
Tests cardinal rotation, tilt and scale, which the real sheets do not cover. No API. 'robust' adds a wide search to the recipe."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg
import truth_seed as ts

truth = json.load(open(reg.OUT / "truth/truth.json"))
CASES = {"rot90": (1, 90), "rot180": (1, 180), "rot270": (1, 270), "tilt5": (1, 5), "tilt-10": (1, -10), "scale0.6": (0.6, 0), "scale0.75": (0.75, 0), "scale1.3": (1.3, 0), "scale1.6": (1.6, 0)}


def perturb(im, s, deg):
    h, w = im.shape[:2]
    R = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, s)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)
    q = corners @ R.T
    mn, mx = q.min(0), q.max(0)
    R[:, 2] -= mn
    size = (int(np.ceil(mx[0] - mn[0])), int(np.ceil(mx[1] - mn[1])))
    return cv2.warpAffine(im, R, size, flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderValue=(255, 255, 255)), np.vstack([R, [0, 0, 1]])


def wide(c):
    """Cardinal orientations, then scale 0.5-2 and tilt up to 12 degrees, each scored by the coarse NCC."""
    best = (-2, None)
    h, w = c["mov_soft"].shape
    for k in range(4):
        def rot(a):
            return np.rot90(a, -k).copy()
        # rot90 by -k, clockwise k quarter turns
        cc = dict(c, mov_ink=rot(c["mov_ink"]))
        Rk = {0: np.eye(3), 1: np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1.0]]), 2: np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1.0]]), 3: np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1.0]])}[k]
        b = reg.apply(Rk, [c["mov_box"][:2], c["mov_box"][2:]])
        x1, y1, x2, y2 = int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 0].max()), int(b[:, 1].max())
        cc["mov_box"] = (x1, y1, x2, y2)
        v, H = coarse_scored(cc, np.geomspace(0.5, 2.0, 25), range(-12, 13, 3))
        if H is not None and v > best[0]:
            best = (v, H @ Rk)
    return best[1]


def coarse_scored(c, scales, angles):
    d = {}
    H = reg.coarse(c, scales=scales, angles=angles, score=d)
    return d.get("v", -2), H


rows = []
pairs = [("bme_f", 2), ("bme_k", 2), ("schenley", 2), ("habs_va_bldg1", 2), ("mml_capitol", 0), ("nibs_clinic", 1), ("wfu_efird", 0), ("habs_stelizabeths_center", 2)]
for bid, f in pairs:
    ref = ts.REF[bid]
    pts = [(np.array(v["floors"][str(f)]["xy"], float), np.array(v["ref"], float)) for v in truth[bid].values() if str(f) in v["floors"]]
    a, b = reg.sheet(bid, f), reg.sheet(bid, ref)
    width = b.shape[1]
    for name, (s, deg) in CASES.items():
        am, T = perturb(a, s, deg)
        box = reg.plan_box(bid, f, a.shape)
        cn = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], float)
        q = reg.apply(T, cn)
        c = {"mov_ink": reg.ink(am), "ref_ink": reg.ink(b), "mov_box": (int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max())), "ref_box": reg.plan_box(bid, ref, b.shape)}
        c["mov_soft"], c["ref_soft"] = cv2.GaussianBlur(c["mov_ink"], (0, 0), 1.2), cv2.GaussianBlur(c["ref_ink"], (0, 0), 1.2)
        for method in ("recipe", "robust"):
            try:
                H = reg.blocks(c) if method == "recipe" else reg.blocks(c, H0=wide(c))
                e = np.median(np.linalg.norm(reg.apply(H @ T, [p[0] for p in pts]) - np.array([p[1] for p in pts]), axis=1)) * 100 / width
            except (cv2.error, TypeError, ValueError) as ex:
                e = float("nan")
            rows.append({"bid": bid, "case": name, "method": method, "pct": round(float(e), 3)})
    print(bid, "done", flush=True)
json.dump(rows, open(reg.OUT / "synth.json", "w"), indent=1)
print(f"{'case':10s}{'recipe <1%':>12s}{'robust <1%':>12s}{'robust med%':>13s}")
for name in CASES:
    a = [r["pct"] for r in rows if r["case"] == name and r["method"] == "recipe"]
    b = [r["pct"] for r in rows if r["case"] == name and r["method"] == "robust"]
    print(f"{name:10s}{sum(x < 1 for x in a):8d}/{len(a)}{sum(x < 1 for x in b):10d}/{len(b)}{np.nanmedian(b):13.2f}")
