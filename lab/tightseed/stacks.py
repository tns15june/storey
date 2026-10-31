"""Aligned frames for every stack with the reg.py recipe. No API: plan box is the ink extent, or a known one, not Spark.
Writes data/lab/tightseed/stacks/<stack>/f<i>.png, H.json (source frame px -> stack px) and overlay.jpg."""
import warnings; warnings.filterwarnings("ignore")
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg
ROOT = Path("/Users/tsharma/Github/Meta"); OUT = ROOT / "data/lab/tightseed/stacks"; AUD = ROOT / "data/lab/audit"


def ink_box(ink, pct=(0.3, 99.7), pad=12):
    ys, xs = np.nonzero(ink)
    x1, x2 = np.percentile(xs, pct); y1, y2 = np.percentile(ys, pct)
    h, w = ink.shape
    return max(0, int(x1 - pad)), max(0, int(y1 - pad)), min(w, int(x2 + pad)), min(h, int(y2 + pad))


def ctx(mov, ref, mbox=None, rbox=None):
    a, b = reg.ink(mov), reg.ink(ref)
    c = dict(mov_ink=a, ref_ink=b, mov_box=mbox or ink_box(a), ref_box=rbox or ink_box(b))
    c["mov_soft"], c["ref_soft"] = cv2.GaussianBlur(a, (0, 0), 1.2), cv2.GaussianBlur(b, (0, 0), 1.2)
    return c


def align_stack(ims, ref, boxes=None, model="similarity"):
    """Each floor onto the ref floor. Returns Hs and inlier counts."""
    Hs, info = {}, {}
    for i, im in enumerate(ims):
        if i == ref:
            Hs[i] = np.eye(3); info[i] = dict(inliers=None, matches=None); continue
        c = ctx(im, ims[ref], boxes and boxes[i], boxes and boxes[ref])
        Hs[i] = reg.blocks(c, model=model)
        info[i] = dict(inliers=c.get("inliers"), matches=c.get("matches"))
    return Hs, info


def overlay(ims, Hs, ref, path, W=900):
    """Ref ink red, floor ink cyan. Black means they agree."""
    shape = ims[ref].shape
    rows = []
    r = reg.ink(ims[ref]) > 0
    for i, im in enumerate(ims):
        m = reg.ink(reg.warp(im, Hs[i], shape, 255)) > 0
        v = np.full(shape, 255, np.uint8)
        v[r] = (60, 60, 255)
        v[m] = (255, 200, 0)
        v[r & m] = (0, 0, 0)
        s = W / shape[1]
        v = cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.putText(v, f"f{i}", (8, 30), 0, 1, (0, 0, 255), 2)
        rows.append(v)
    n = len(rows); cols = 2
    while len(rows) % cols:
        rows.append(np.full_like(rows[0], 255))
    cv2.imwrite(str(path), np.vstack([np.hstack(rows[k:k + cols]) for k in range(0, len(rows), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def save(name, ims, Hs, ref, info):
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(ims):
        cv2.imwrite(str(d / f"f{i}.png"), reg.warp(im, Hs[i], ims[ref].shape, 255))
    json.dump(dict(ref=ref, H={str(k): v.tolist() for k, v in Hs.items()}, info={str(k): v for k, v in info.items()}), open(d / "H.json", "w"), indent=1)
    overlay(ims, Hs, ref, d / "overlay.jpg")
    print(name, ims[ref].shape, info, flush=True)


if __name__ == "__main__":
    for name, n in (("bme_k", 4), ("habs_va_bldg1", 5), ("wfu_huffman", 4)):
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        ims = [cv2.imread(str(AUD / name / f"f{i}.png")) for i in range(n)]
        Hs, info = align_stack(ims, 1)
        save(name, ims, Hs, 1, info)
