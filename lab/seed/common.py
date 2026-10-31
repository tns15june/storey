"""Shared bits for the seed experiment: load catalog floors as 2000 px sheets, align them, make a stack video."""
import json, subprocess, warnings
from pathlib import Path
import cv2, numpy as np
warnings.filterwarnings("ignore")
import fitz

ROOT = Path("/Users/tsharma/Github/Meta")
OUT = ROOT / "data/lab/seed"
PLANS = ROOT / "data/plans"
W = 2000
CAT = {x["id"]: x for x in json.load(open(PLANS / "catalog.json"))}


def load_floor(bid, f):
    p = PLANS / bid / f["file"]
    if p.suffix.lower() == ".pdf":
        pg = fitz.open(p)[f["page"]]
        px = pg.get_pixmap(matrix=fitz.Matrix(W / pg.rect.width, W / pg.rect.width), alpha=False)
        return cv2.cvtColor(np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, 3), cv2.COLOR_RGB2BGR)
    im = cv2.imread(str(p))
    return cv2.resize(im, (W, round(im.shape[0] * W / im.shape[1])), interpolation=cv2.INTER_AREA)


def _ink(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 18)


def overlap(mov, ref):
    """Share of ink in `mov` that lies within 3 px of ink in `ref`. 1 is a perfect match."""
    a, b = _ink(mov) > 0, _ink(ref) > 0
    d = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    return float((d[a] <= 3).mean()) if a.any() else 0.0


def align(mov, ref):
    """Warp `mov` onto `ref`. Tries identity, ORB similarity and ink phase correlation, keeps the best overlap."""
    h, w = ref.shape[:2]
    pad = (235, 235, 235)
    cand = {"identity": np.float32([[1, 0, 0], [0, 1, 0]])}
    orb = cv2.ORB_create(12000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    src, dst = np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kb[x.trainIdx].pt for x in m])
    A, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6, maxIters=20000)
    if A is not None:
        cand["orb"] = np.float32(A)
    size = (w, h)
    ib = cv2.GaussianBlur(_ink(ref).astype(np.float32), (0, 0), 4)
    im = _ink(mov)
    F = np.conj(np.fft.rfft2(ib - ib.mean()))
    for scale in np.arange(0.95, 1.051, 0.01):
        S = np.float32([[scale, 0, 0], [0, scale, 0]])
        ia = cv2.GaussianBlur(cv2.warpAffine(im, S, size).astype(np.float32), (0, 0), 4)
        c = np.fft.irfft2(np.fft.rfft2(ia - ia.mean()) * F, s=ia.shape)
        dy, dx = np.unravel_index(np.argmax(c), c.shape)
        dy, dx = (dy - h if dy > h / 2 else dy), (dx - w if dx > w / 2 else dx)
        cand[f"xcorr{scale:.2f}"] = np.float32([[scale, 0, -dx], [0, scale, -dy]])
    score = lambda A: overlap(cv2.warpAffine(mov, A, size, borderValue=pad), ref)
    k = max(cand, key=lambda k: score(cand[k]))
    A = cand[k].copy()
    try:  # shortcut: ECC only polishes, a failure keeps the coarse answer
        B = cv2.vconcat([A, np.float32([[0, 0, 1]])])[:2].astype(np.float32)
        _, B = cv2.findTransformECC(ib, cv2.GaussianBlur(im.astype(np.float32), (0, 0), 4), cv2.invertAffineTransform(B).astype(np.float32),
                                    cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-5), None, 1)
        B = cv2.invertAffineTransform(B).astype(np.float32)
        if score(B) > score(A):
            A, k = B, k + "+ecc"
    except cv2.error:
        pass
    return cv2.warpAffine(mov, A, size, borderValue=pad), {"method": k, "overlap": round(score(A), 3), "A": A.tolist()}


def stack_mp4(frames, path, repeat=4):
    """Write frames as an H.264 clip, each repeated. Returns (w, h) of the video."""
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}", "-r", "4", "-i", "-",
                          "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        for _ in range(repeat):
            p.stdin.write(f[:h2, :w2].tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return w2, h2


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.0


def hit(box, t, grow=0.15):
    """Does `box` sit on truth box `t`: IoU of at least 0.25, or its centre inside t grown by 15%, with no more than 4x t's area."""
    gx, gy = (t[2] - t[0]) * grow, (t[3] - t[1]) * grow
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    inside = t[0] - gx <= cx <= t[2] + gx and t[1] - gy <= cy <= t[3] + gy
    big = (box[2] - box[0]) * (box[3] - box[1]) > 4 * (t[2] - t[0]) * (t[3] - t[1])
    return iou(box, t) >= 0.25 or (inside and not big)


def to_px(o, w, h):
    x1, x2 = sorted((o["x_min"], o["x_max"]))
    y1, y2 = sorted((o["y_min"], o["y_max"]))
    return [min(max(round(x1 * w / 1000), 0), w - 1), min(max(round(y1 * h / 1000), 0), h - 1), min(max(round(x2 * w / 1000), 0), w - 1), min(max(round(y2 * h / 1000), 0), h - 1)]


TRUTH = json.load(open(ROOT / "lab/seed/truth.json"))
