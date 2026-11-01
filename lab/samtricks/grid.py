"""Labelling aid: a sheet or sheet crop with a 0-1000 grid drawn over it. No API calls."""
import sys

import cv2

sys.path.insert(0, "lab/samtricks")
import reg


def grid(key, box=(0, 0, 1000, 1000), step=100, zoom=1.0, name=None):
    im = reg.sheet(key)
    h, w = im.shape[:2]
    x0, y0, x1, y1 = box
    px = lambda v, n: int(round(v / 1000 * n))
    c = im[px(y0, h):px(y1, h), px(x0, w):px(x1, w)].copy()
    if zoom != 1:
        c = cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    H, W = c.shape[:2]
    sx, sy = W / (x1 - x0), H / (y1 - y0)
    for v in range((x0 // step) * step, x1 + 1, step):
        if v < x0:
            continue
        X = int((v - x0) * sx)
        cv2.line(c, (X, 0), (X, H), (0, 0, 255), 1)
        for Y in range(14, H, max(int(H / 3), 200)):
            cv2.putText(c, str(v), (X + 2, Y), 0, .45, (0, 0, 255), 1, cv2.LINE_AA)
    for v in range((y0 // step) * step, y1 + 1, step):
        if v < y0:
            continue
        Y = int((v - y0) * sy)
        cv2.line(c, (0, Y), (W, Y), (200, 0, 0), 1)
        for X in range(2, W, max(int(W / 3), 300)):
            cv2.putText(c, str(v), (X, Y - 2), 0, .45, (200, 0, 0), 1, cv2.LINE_AA)
    d = reg.OUT / "grid"
    d.mkdir(exist_ok=True)
    p = d / f"{name or key}.jpg"
    cv2.imwrite(str(p), c, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return p


if __name__ == "__main__":
    for k in dict.fromkeys(reg.DEV + reg.HELD + reg.WORD_SHEETS):
        print(grid(k))
