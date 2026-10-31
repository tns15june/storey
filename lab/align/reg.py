"""Sheet-to-sheet registration methods. Each returns a 3x3 matrix mapping mov sheet px to ref sheet px, or None."""
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path("/Users/tsharma/Github/Meta/data/lab/align")
SPARK = json.load(open(OUT / "spark_sheets.json"))
INDEX = json.load(open(OUT / "sheets/index.json"))


def sheet(bid, i):
    return cv2.imread(str(OUT / f"sheets/{bid}/{i}.jpg"))


def plan_box(bid, i, shape, pad=0.02):
    b = SPARK[f"{bid}/{i}"]["read"]["plan_box"]
    h, w = shape[:2]
    p = pad * max(w, h)
    return (max(0, int(b["x1"] * w / 1000 - p)), max(0, int(b["y1"] * h / 1000 - p)), min(w, int(b["x2"] * w / 1000 + p)), min(h, int(b["y2"] * h / 1000 + p)))


def ink(im, thresh=0.72):
    """Paper-normalised ink map, 0..1, so tint, stains and film edges do not count as structure."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.medianBlur(cv2.resize(g, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA).astype(np.uint8), 15)
    bg = cv2.resize(bg.astype(np.float32), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR) + 1
    return (g / bg < thresh).astype(np.float32)


def masked(a, box):
    out = np.zeros_like(a)
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = a[y1:y2, x1:x2]
    return out


def h3(A):
    return np.vstack([A, [0, 0, 1]]).astype(np.float64)


def sim(s, theta_deg, tx, ty):
    t = np.deg2rad(theta_deg)
    c, d = s * np.cos(t), s * np.sin(t)
    return np.array([[c, -d, tx], [d, c, ty], [0, 0, 1]], float)


def apply(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


def warp(im, H, shape, border=255):
    return cv2.warpPerspective(im, H, (shape[1], shape[0]), flags=cv2.INTER_AREA if H[0, 0] < 0.8 else cv2.INTER_LINEAR, borderValue=border if im.ndim == 2 else (border,) * 3)


# ---------- baselines ----------
def identity(c):
    return np.eye(3)


def _orb(a, b, ma=None, mb=None, n=8000):
    orb = cv2.ORB_create(n)
    ka, da = orb.detectAndCompute(a, ma)
    kb, db = orb.detectAndCompute(b, mb)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 8:
        return None
    src, dst = np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kb[x.trainIdx].pt for x in m])
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6, maxIters=20000)
    return None if A is None else h3(A)


def orb_global(c):
    return _orb(c["mov_g"], c["ref_g"])


def _boxmask(shape, box):
    m = np.zeros(shape[:2], np.uint8)
    m[box[1]:box[3], box[0]:box[2]] = 255
    return m


def orb_mask(c):
    return _orb(c["mov_g"], c["ref_g"], _boxmask(c["mov_g"].shape, c["mov_box"]), _boxmask(c["ref_g"].shape, c["ref_box"]))


def bbox(c):
    """Rough: centre the Spark plan boxes, scale from the mean box-side ratio."""
    a, b = c["mov_box"], c["ref_box"]
    s = 0.5 * ((b[2] - b[0]) / (a[2] - a[0]) + (b[3] - b[1]) / (a[3] - a[1]))
    ca, cb = np.array([a[0] + a[2], a[1] + a[3]]) / 2, np.array([b[0] + b[2], b[1] + b[3]]) / 2
    t = cb - s * ca
    return sim(s, 0, t[0], t[1])


def bbox_translate(c):
    a, b = c["mov_box"], c["ref_box"]
    ca, cb = np.array([a[0] + a[2], a[1] + a[3]]) / 2, np.array([b[0] + b[2], b[1] + b[3]]) / 2
    return sim(1, 0, *(cb - ca))


def phase(c):
    """Phase correlation on blurred masked ink, translation only."""
    a = cv2.GaussianBlur(masked(c["mov_ink"], c["mov_box"]), (0, 0), 3)
    b = cv2.GaussianBlur(masked(c["ref_ink"], c["ref_box"]), (0, 0), 3)
    h = max(a.shape[0], b.shape[0]); w = max(a.shape[1], b.shape[1])
    pa, pb = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    pa[:a.shape[0], :a.shape[1]] = a
    pb[:b.shape[0], :b.shape[1]] = b
    (dx, dy), _ = cv2.phaseCorrelate(pa, pb, cv2.createHanningWindow((w, h), cv2.CV_32F))
    return sim(1, 0, dx, dy)


def coarse(c, scales=np.linspace(0.84, 1.2, 19), angles=(-2, -1, 0, 1, 2), f=5, score=None):
    """Plan-box crop of mov searched over scale and small rotation inside ref ink, by NCC at 1/f resolution."""
    a = cv2.GaussianBlur(cv2.resize(masked(c["mov_ink"], c["mov_box"]), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    b = cv2.GaussianBlur(cv2.resize(masked(c["ref_ink"], c["ref_box"]), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    x1, y1, x2, y2 = [v / f for v in c["mov_box"]]
    best = (-2, None)
    pad = 200
    B = cv2.copyMakeBorder(b, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for s in scales:
        for th in angles:
            Rm = cv2.getRotationMatrix2D((cx, cy), th, s)
            w, h = a.shape[1], a.shape[0]
            r = cv2.warpAffine(a, Rm, (w, h))
            px = int((x2 - x1) * s / 2 + 2); py = int((y2 - y1) * s / 2 + 2)
            xa, xb, ya, yb = max(0, int(cx) - px), min(w, int(cx) + px), max(0, int(cy) - py), min(h, int(cy) + py)
            t = r[ya:yb, xa:xb]
            if t.shape[0] >= B.shape[0] or t.shape[1] >= B.shape[1]:
                continue
            res = cv2.matchTemplate(B, t, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(res)
            if v > best[0]:
                best = (v, (s, th, loc[0] - pad - xa, loc[1] - pad - ya, Rm))
    if best[1] is None:
        return None
    v, (s, th, dx, dy, Rm) = best
    if score is not None:
        score["v"] = v
    H = np.eye(3)
    H[:2] = Rm
    H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]]) @ H
    S = np.diag([f, f, 1.0])
    return S @ H @ np.linalg.inv(S)


def matches(c, H0, patch, step, search, min_ncc, scale=1.0):
    """NCC block matches: ref point -> mov point. Blocks with little ink are skipped; RANSAC handles walls that moved."""
    mw = warp(c["mov_soft"], H0, c["ref_soft"].shape, 0)
    ref = c["ref_soft"]
    if scale != 1.0:
        mw = cv2.resize(mw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ref = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = [int(v * scale) for v in c["ref_box"]]
    src, dst, w = [], [], []
    p = patch // 2
    for y in range(y1 + p, y2 - p, step):
        for x in range(x1 + p, x2 - p, step):
            t = ref[y - p:y + p, x - p:x + p]
            if not 0.03 < (t > 0.2).mean() < 0.7:
                continue
            ya, yb, xa, xb = y - p - search, y + p + search, x - p - search, x + p + search
            if ya < 0 or xa < 0 or yb > mw.shape[0] or xb > mw.shape[1]:
                continue
            r = cv2.matchTemplate(mw[ya:yb, xa:xb], t, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(r)
            if v < min_ncc:
                continue
            src.append((xa + loc[0] + p, ya + loc[1] + p))
            dst.append((x, y))
            w.append(v)
    if len(src) < 6:
        return None, None
    return apply(np.linalg.inv(H0), np.array(src, float) / scale), np.array(dst, float) / scale


def blocks(c, H0=None, model="similarity"):
    """Coarse init, wide block match at half scale, then a tight full-scale pass. Keeps the fit with the most inliers."""
    H = coarse(c) if H0 is None else H0
    for patch, step, search, ncc, scale, thr in ((64, 24, 40, 0.35, 0.5, 8), (96, 40, 14, 0.45, 1.0, 3)):
        s, d = matches(c, H, patch, step, search, ncc, scale)
        if s is None:
            return H
        if model == "homography" and scale == 1.0:
            M, inl = cv2.findHomography(s, d, cv2.RANSAC, thr, maxIters=20000)
            H2 = M
        elif model == "affine" and scale == 1.0:
            A, inl = cv2.estimateAffine2D(s, d, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=20000)
            H2 = None if A is None else h3(A)
        else:
            A, inl = cv2.estimateAffinePartial2D(s, d, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=20000)
            H2 = None if A is None else h3(A)
        if H2 is None or inl.sum() < 8:
            return H
        H = H2
    c["inliers"] = int(inl.sum()); c["matches"] = len(s)
    return H


def blocks_affine(c):
    return blocks(c, model="affine")


def blocks_homography(c):
    return blocks(c, model="homography")


def ecc(c, H0=None, motion=cv2.MOTION_EUCLIDEAN):
    """ECC on blurred masked ink from the coarse init. Euclidean by default; scale comes from the init."""
    H = coarse(c) if H0 is None else H0
    f = 2
    S = np.diag([1 / f, 1 / f, 1.0])
    a = cv2.GaussianBlur(cv2.resize(c["ref_soft"], None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA), (0, 0), 3)
    mw = warp(c["mov_soft"], H, c["ref_soft"].shape, 0)
    b = cv2.GaussianBlur(cv2.resize(mw, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA), (0, 0), 3)
    mask = cv2.resize(_boxmask(c["ref_soft"].shape, c["ref_box"]), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        _, W = cv2.findTransformECC(a, b, W, motion, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), mask, 5)
    except cv2.error:
        return H
    # W maps ref -> warped mov, at 1/f scale
    return np.linalg.inv(np.linalg.inv(S) @ h3(W) @ S) @ H


def ecc_affine(c):
    return ecc(c, motion=cv2.MOTION_AFFINE)


def outline(c):
    """Filled building footprint, ECC on its soft edge. Footprints differ floor to floor, so expect it to be rough."""
    def foot(ink_map, box):
        m = masked(ink_map, box)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        cl = cv2.morphologyEx((m > 0).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k)
        cnt, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnt:
            return cl
        f = np.zeros_like(cl)
        cv2.drawContours(f, [max(cnt, key=cv2.contourArea)], -1, 255, -1)
        return cv2.GaussianBlur(f, (0, 0), 6).astype(np.float32) / 255
    cc = dict(c, mov_soft=foot(c["mov_ink"], c["mov_box"]), ref_soft=foot(c["ref_ink"], c["ref_box"]))
    H0 = bbox(c)
    return ecc(cc, H0=H0, motion=cv2.MOTION_AFFINE)


def context(bid, mov, ref):
    a, b = sheet(bid, mov), sheet(bid, ref)
    c = {"mov_g": cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), "ref_g": cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
         "mov_ink": ink(a), "ref_ink": ink(b), "mov_box": plan_box(bid, mov, a.shape), "ref_box": plan_box(bid, ref, b.shape)}
    c["mov_soft"] = cv2.GaussianBlur(c["mov_ink"], (0, 0), 1.2)
    c["ref_soft"] = cv2.GaussianBlur(c["ref_ink"], (0, 0), 1.2)
    return c


METHODS = {"identity": identity, "orb_global": orb_global, "orb_mask": orb_mask, "bbox_translate": bbox_translate, "bbox": bbox, "phase": phase,
           "coarse": coarse, "outline": outline, "ecc": ecc, "ecc_affine": ecc_affine, "blocks": blocks, "blocks_affine": blocks_affine, "blocks_homography": blocks_homography}

if __name__ == "__main__":
    # Self-check: a known similarity applied to a real sheet must come back within 2 px at the plan-box corners.
    bid = "bme_f"
    im = sheet(bid, 1)
    T = sim(0.93, 1.5, 40, -25)
    moved = cv2.warpPerspective(im, T, (im.shape[1], im.shape[0]), borderValue=(255, 255, 255))
    ref_box = plan_box(bid, 1, im.shape)
    cn = np.array([[ref_box[0], ref_box[1]], [ref_box[2], ref_box[3]]], float)
    c = {"mov_ink": ink(moved), "ref_ink": ink(im), "ref_box": ref_box,
         "mov_box": tuple(np.r_[apply(T, cn).min(0) - 10, apply(T, cn).max(0) + 10].astype(int)), "mov_g": None, "ref_g": None}
    c["mov_soft"], c["ref_soft"] = cv2.GaussianBlur(c["mov_ink"], (0, 0), 1.2), cv2.GaussianBlur(c["ref_ink"], (0, 0), 1.2)
    H = blocks(c)
    err = np.linalg.norm(apply(H @ T, cn) - cn, axis=1)
    print("self-check error px", err.round(2), c.get("inliers"), c.get("matches"))
    assert err.max() < 2
