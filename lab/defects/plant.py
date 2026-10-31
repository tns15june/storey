"""Plant defects by editing ink, never by pasting flat boxes: paper grain outside the ink stays untouched."""
import sys
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/defects')
from common import *


def paper_of(img, x1, y1, x2, y2, k=31):
    """Local paper estimate: closing removes dark strokes narrower than k."""
    pad = k + 8
    X1, Y1, X2, Y2 = max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, img.shape[1]), min(y2 + pad, img.shape[0])
    c = img[Y1:Y2, X1:X2]
    bg = cv2.morphologyEx(c, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 8)
    return bg, (X1, Y1, X2, Y2)


def erase(img, x1, y1, x2, y2, region=None, seed=0):
    """Remove ink inside the box (or the bool `region`). Paper tone comes from normalised convolution over nearby non-ink pixels, grain from real paper residuals."""
    rng = np.random.default_rng(seed)
    bg0, (X1, Y1, X2, Y2) = paper_of(img, x1, y1, x2, y2)
    c = img[Y1:Y2, X1:X2].astype(np.float32)
    d0 = (bg0.astype(np.float32) - c).mean(axis=2)
    isp = np.float32(cv2.erode(np.uint8(d0 < 10), np.ones((5, 5), np.uint8)))
    if isp.sum() < 100:
        isp = np.float32(d0 < 25)
    num = cv2.GaussianBlur(c * isp[..., None], (0, 0), 14)
    den = cv2.GaussianBlur(isp, (0, 0), 14)[..., None] + 1e-3
    wt = den / (den + .12)  # in dense ink there is no nearby paper, so lean on the sheet's own paper tone
    bg = wt * num / den + (1 - wt) * np.median(img.reshape(-1, 3), axis=0).astype(np.float32)
    gray_d = (bg - c).mean(axis=2)
    pap = (c - bg)[isp > 0]
    pap = pap[np.abs(pap).max(axis=1) < np.percentile(np.abs(pap).max(axis=1), 98) + 1e-3]
    m = np.zeros(c.shape[:2], np.uint8)
    if region is None:
        m[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = 1
    else:
        m[:] = region[Y1:Y2, X1:X2]
    ink = (gray_d > 6) & (m > 0)
    ink = cv2.dilate(np.uint8(ink), np.ones((3, 3), np.uint8)) & cv2.dilate(m, np.ones((3, 3), np.uint8))
    a = cv2.GaussianBlur(np.float32(ink), (0, 0), .7)[..., None]
    fill = bg + pap[rng.integers(0, len(pap), c.shape[:2])]
    out = img.copy()
    out[Y1:Y2, X1:X2] = np.clip(c * (1 - a) + fill * a, 0, 255).astype(np.uint8)
    return out


def ink_color(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = img[g < np.percentile(g, 2)]
    return tuple(float(x) for x in np.median(d, axis=0))


def draw_wall(img, x1, y1, x2, y2, t):
    """A new wall in the sheet's own ink colour with the softness of the scan."""
    out = img.astype(np.float32)
    m = np.zeros(img.shape[:2], np.float32)
    t2 = int(t) // 2
    if y1 == y2:
        cv2.rectangle(m, (x1, y1 - t2), (x2, y1 + t2), 1.0, -1)
    else:
        cv2.rectangle(m, (x1 - t2, y1), (x1 + t2, y2), 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), .8)[..., None]
    col = np.array(ink_color(img), np.float32)
    return np.clip(out * (1 - m) + col * m, 0, 255).astype(np.uint8)


def move(img, x1, y1, x2, y2, dx, dy, region=None, seed=0):
    """Erase what is in the box, then redraw its ink shifted by (dx, dy), darkest pixel wins so lines stay lines."""
    if region is None:
        region = np.zeros(img.shape[:2], bool); region[y1:y2, x1:x2] = True
    patch_src = img.copy()
    out = erase(img, x1, y1, x2, y2, region, seed)
    ys, xs = np.nonzero(region)
    h, w = img.shape[:2]
    ok = (ys + dy >= 0) & (ys + dy < h) & (xs + dx >= 0) & (xs + dx < w)
    ys, xs = ys[ok], xs[ok]
    out[ys + dy, xs + dx] = np.minimum(out[ys + dy, xs + dx], patch_src[ys, xs])
    return out


def rescan(img, seed=0):
    """A fresh scan of the same sheet: small shift and rotation, sensor noise, JPEG."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-.25, .25), 1 + rng.uniform(-.002, .002))
    M[:, 2] += rng.uniform(-5, 5, 2)
    bv = tuple(int(x) for x in np.median(img.reshape(-1, 3), axis=0))
    o = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=bv).astype(np.float32)
    o = np.clip(cv2.GaussianBlur(o, (0, 0), .5) + rng.normal(0, 2.5, o.shape), 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode('.jpg', o, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], cv2.IMREAD_COLOR)


if __name__ == '__main__':
    # self-check: erasing keeps paper pixels byte-identical and removes the ink
    p = np.full((200, 200, 3), (200, 220, 230), np.uint8)
    p = np.clip(p + np.random.default_rng(1).normal(0, 2, p.shape), 0, 255).astype(np.uint8)
    q = p.copy(); q[100, 40:160] = 20
    e = erase(q, 30, 90, 170, 110)
    assert e[100, 100].mean() > 180, e[100, 100]
    assert (e[:80] == q[:80]).all()
    print('plant self-check ok')
