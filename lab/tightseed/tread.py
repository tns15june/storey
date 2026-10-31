"""Tread detector, no model. Short-time 2D FFT: periodic parallel lines put their energy in one narrow peak, single lines and text do not.
`pmap` gives a concentration map; `snap` turns it into a tight box; `score_box` is the post-track check."""
import numpy as np, cv2

SCALES = (24, 40, 64, 96, 128)
CFG = dict(thr=.34, min_std=4.0, pmin=3.0)


def _hann(w):
    h = np.hanning(w + 2)[1:-1]
    return np.outer(h, h)


def _lags(w, H, pmin):
    """Integer lag vectors v, half plane, with v, 2v and v/2 inside the window's reliable range."""
    L = w // 3
    dy, dx = np.mgrid[-L:L + 1, 0:L + 1]
    keep = (np.hypot(dx, dy) >= pmin) & (np.hypot(dx, dy) <= L) & ((dx > 0) | (dy > 0))
    dy, dx = dy[keep], dx[keep]
    return dy, dx, np.rint(dy / 2).astype(int), np.rint(dx / 2).astype(int)


def pmap(gray, cfg=CFG):
    """Q is the windowed autocorrelation at lag v and 2v, less its value at v/2: at least three parallel lines score high, one line or two do not.
    Returns Q, period, angle at 4 px cells, best scale per cell."""
    g = gray.astype(np.float32)
    H, W = g.shape
    S = np.zeros((H // 4 + 1, W // 4 + 1), np.float32); P = np.zeros_like(S); A = np.zeros_like(S)
    for w in SCALES:
        if H < w or W < w:
            continue
        st = max(4, w // 4)
        win = np.lib.stride_tricks.sliding_window_view(g, (w, w))[::st, ::st]
        rows, cols = win.shape[:2]
        pat = win.reshape(-1, w, w)
        std = pat.std((1, 2))
        hw = _hann(w)
        pat = (pat - pat.mean((1, 2), keepdims=True)) * hw
        R = np.fft.irfft2(np.abs(np.fft.rfft2(pat, s=(2 * w, 2 * w))) ** 2, s=(2 * w, 2 * w))
        Hc = np.fft.irfft2(np.abs(np.fft.rfft2(hw, s=(2 * w, 2 * w))) ** 2, s=(2 * w, 2 * w))
        R = R / (R[:, :1, :1] + 1e-6)
        Hc = Hc / Hc[0, 0]
        dy, dx, hy, hx = _lags(w, Hc, cfg["pmin"])
        n = 2 * w
        rv = R[:, dy % n, dx % n] / Hc[dy % n, dx % n]
        r2 = R[:, (2 * dy) % n, (2 * dx) % n] / Hc[(2 * dy) % n, (2 * dx) % n]
        rh = R[:, hy % n, hx % n] / Hc[hy % n, hx % n]
        Q = np.minimum(rv, r2) - np.maximum(rh, 0)
        i = Q.argmax(1)
        q = Q[np.arange(len(Q)), i] * (std >= cfg["min_std"])
        per, ang = np.hypot(dy[i], dx[i]), np.degrees(np.arctan2(dy[i], dx[i])) % 180
        c, per, ang = q.reshape(rows, cols), per.reshape(rows, cols), ang.reshape(rows, cols)
        # each window speaks for its central half, best score wins
        for y in range(rows):
            for x in range(cols):
                y1, x1 = (y * st + w // 4) // 4, (x * st + w // 4) // 4
                y2, x2 = (y * st + 3 * w // 4) // 4 + 1, (x * st + 3 * w // 4) // 4 + 1
                roi = S[y1:y2, x1:x2]; m = roi < c[y, x]
                roi[m] = c[y, x]; P[y1:y2, x1:x2][m] = per[y, x]; A[y1:y2, x1:x2][m] = ang[y, x]
    return S, P, A


def ink(g, p):
    """Lines darker than their neighbourhood. Works on white paper and on grey-filled stair halls."""
    gf = g.astype(np.float32)
    return (cv2.GaussianBlur(gf, (0, 0), max(2.0, 1.5 * p)) - gf) > 8


def modes(S, P, A, thr, top=3):
    """Dominant (period, angle) among cells over thr: 15 degree and half-octave bins, each mode needs a fifth of the cells."""
    ok = S >= thr
    if ok.sum() < 8:
        return []
    ab = np.round(A[ok] / 15).astype(int) % 12
    pb = np.round(np.log2(np.maximum(P[ok], 1)) * 2).astype(int)
    keys, counts = np.unique(ab * 100 + pb, return_counts=True)
    out = []
    for k in keys[np.argsort(-counts)][:top]:
        c = (ab * 100 + pb) == k
        if c.sum() >= .2 * ok.sum():
            out.append((float(np.median(P[ok][c])), float(np.median(A[ok][c])), int(c.sum())))
    return out


def comb(g, p, ang):
    """Ink with ink one period away on both sides and paper half a period away: true on tread lines, false on walls, text and single lines."""
    I = ink(g, p).astype(np.uint8)
    D = cv2.dilate(I, np.ones((3, 3), np.uint8))
    dx, dy = p * np.cos(np.radians(ang)), p * np.sin(np.radians(ang))
    def sh(a, tx, ty):
        return cv2.warpAffine(a, np.float32([[1, 0, tx], [0, 1, ty]]), (a.shape[1], a.shape[0]), flags=cv2.INTER_NEAREST)
    c = I & sh(D, dx, dy) & sh(D, -dx, -dy)
    if p >= 5:  # below that a stroke fills the half period
        c = c & (1 - sh(I, dx / 2, dy / 2)) & (1 - sh(I, -dx / 2, -dy / 2))
    return c


def tight(g, cfg=CFG):
    """Comb masks of the dominant modes, joined across the gaps between lines. Returns (mask, modes)."""
    S, P, A = pmap(g, cfg)
    ms = modes(S, P, A, cfg["thr"])
    if not ms:
        return None, ms
    M = np.zeros(g.shape, np.uint8)
    for p, ang, _ in ms:
        c = comb(g, p, ang)
        k = int(round(p)) + 3
        M |= cv2.morphologyEx(c, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return M, ms


def snap(gray_frame, box, cfg=CFG, grow=.15):
    """Tight rectangle of tread runs inside a loose box. Returns (box, info) or (None, info) when nothing periodic is found."""
    x1, y1, x2, y2 = box
    gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
    X1, Y1, X2, Y2 = max(0, int(x1 - gx)), max(0, int(y1 - gy)), min(gray_frame.shape[1], int(x2 + gx)), min(gray_frame.shape[0], int(y2 + gy))
    M, ms = tight(gray_frame[Y1:Y2, X1:X2], cfg)
    if M is None or not M.any():
        return None, dict(modes=[])
    p = ms[0][0]
    n, lab, st, _ = cv2.connectedComponentsWithStats(M, connectivity=8)
    if n < 2:
        return None, dict(modes=ms)
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= max(.25 * big, (3 * p) ** 2)]
    if not keep:
        return None, dict(modes=ms)
    xs = [st[i, 0] for i in keep]; ys = [st[i, 1] for i in keep]
    xe = [st[i, 0] + st[i, 2] for i in keep]; ye = [st[i, 1] + st[i, 3] for i in keep]
    # the comb needs a neighbour on both sides, so the outermost lines are one period outside the mask
    ex, ey = p * abs(np.cos(np.radians(ms[0][1]))), p * abs(np.sin(np.radians(ms[0][1])))
    return [float(min(xs) + X1 - ex), float(min(ys) + Y1 - ey), float(max(xe) + X1 + ex), float(max(ye) + Y1 + ey)], dict(modes=ms, runs=len(keep), period=p, angle=ms[0][1])


def score_box(gray_frame, box, cfg=CFG, pad=8):
    """Check after tracking. Share of the box that is comb mask, and the Q score behind it."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(gray_frame.shape[1], x2 + pad), min(gray_frame.shape[0], y2 + pad)
    if Y2 - Y1 < SCALES[0] or X2 - X1 < SCALES[0]:
        return dict(cover=0.0, q90=0.0, period=0.0, steps=0)
    g = gray_frame[Y1:Y2, X1:X2]
    S, P, A = pmap(g, cfg)
    q90 = float(np.percentile(S, 90))
    M, ms = tight(g, cfg)
    return dict(cover=0.0 if M is None else float(M.mean()), q90=q90, period=ms[0][0] if ms else 0.0, steps=steps(M, ms))


def steps(M, ms):
    """Most treads in one connected run: extent across the lines over the period, plus the two outer lines the comb cannot mark."""
    if M is None or not ms:
        return 0
    n, lab = cv2.connectedComponents(M, connectivity=8)
    best = 0
    for p, ang, _ in ms:
        ux, uy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        for i in range(1, n):
            ys, xs = np.nonzero(lab == i)
            if len(xs) < 4 * p:
                continue
            t = xs * ux + ys * uy
            w = xs * -uy + ys * ux
            if np.ptp(w) >= 2.5 * p:  # treads are longer than they are deep
                best = max(best, int(np.ptp(t) / p) + 2)
    return best


if __name__ == "__main__":
    # self-check on a synthetic drawing: nine treads must be found and snapped, a lone rectangle must not
    im = np.full((300, 300), 235, np.uint8)
    cv2.rectangle(im, (20, 20), (280, 280), 40, 2)
    for x in range(100, 200, 10):
        cv2.line(im, (x, 110), (x, 190), 40, 1)
    b, info = snap(im, (40, 40, 260, 260))
    assert b is not None and abs(b[0] - 100) < 20 and abs(b[2] - 195) < 20 and abs(b[1] - 110) < 20 and abs(b[3] - 190) < 20, (b, info)
    assert 8 <= info["period"] <= 12, info
    plain = np.full((300, 300), 235, np.uint8); cv2.rectangle(plain, (80, 80), (220, 220), 40, 2)
    r = score_box(plain, (60, 60, 240, 240))
    assert r["cover"] < .05, r
    wall = np.full((300, 300), 235, np.uint8); wall[100:200, 100:200] = 30
    assert score_box(wall, (60, 60, 240, 240))["cover"] < .05
    print("tread self-check ok", b, info)
