"""Label and overlay views. `crop` shows one stack region across floors with a pixel grid; `blend` overlays two floors. No API."""
import sys, json
from pathlib import Path
import cv2, numpy as np
ROOT = Path("/Users/tsharma/Github/Meta"); S = ROOT / "data/lab/tightseed/stacks"; V = ROOT / "data/lab/tightseed/views"; V.mkdir(exist_ok=True)


def frames(name):
    fs = sorted(S.joinpath(name).glob("f*.png"), key=lambda p: int(p.stem[1:]))
    return [cv2.imread(str(p)) for p in fs]


def grid_crop(im, box, zoom, step=20, boxes=()):
    """Crop at zoom with tick lines every `step` source px. Each entry of boxes is (box, bgr, label)."""
    x1, y1, x2, y2 = box
    c = cv2.resize(im[y1:y2, x1:x2], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    for x in range((x1 // step + 1) * step, x2, step):
        X = int((x - x1) * zoom); major = x % (step * 5) == 0
        cv2.line(c, (X, 0), (X, c.shape[0] if major else 6), (255, 0, 255) if major else (200, 120, 0), 1)
        if major: cv2.putText(c, str(x), (X + 2, 12), 0, .4, (255, 0, 255), 1)
    for y in range((y1 // step + 1) * step, y2, step):
        Y = int((y - y1) * zoom); major = y % (step * 5) == 0
        cv2.line(c, (0, Y), (c.shape[1] if major else 6, Y), (255, 0, 255) if major else (200, 120, 0), 1)
        if major: cv2.putText(c, str(y), (2, Y - 2), 0, .4, (255, 0, 255), 1)
    for b, col, lab in boxes:
        p = [int((b[0] - x1) * zoom), int((b[1] - y1) * zoom), int((b[2] - x1) * zoom), int((b[3] - y1) * zoom)]
        cv2.rectangle(c, (p[0], p[1]), (p[2], p[3]), col, 1)
        if lab: cv2.putText(c, lab, (p[0] + 2, max(10, p[1] - 3)), 0, .4, col, 1)
    return c


def strip(name, boxes, zoom=3, step=20, cols=None, out=None, marks=None):
    """Rows of regions, one column per floor."""
    fr = frames(name); rows = []
    for b in boxes:
        tiles = []
        for i, im in enumerate(fr):
            m = (marks or {}).get(i, [])
            t = grid_crop(im, b, zoom, step, m)
            cv2.putText(t, f"f{i}", (t.shape[1] - 30, t.shape[0] - 6), 0, .6, (0, 0, 255), 2)
            tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 180, 0)))
        while cols and len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        rows.append(np.hstack(tiles) if not cols else np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]))
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    p = V / (out or f"{name}_strip.jpg")
    cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return p


def blend(name, a, b, box=None, scale=1.0, out=None):
    fr = frames(name)
    g = [cv2.cvtColor(fr[i], cv2.COLOR_BGR2GRAY) for i in (a, b)]
    v = np.dstack([g[1], g[1], g[0]])  # floor a shows red-ish where only a has ink removed
    if box: v = v[box[1]:box[3], box[0]:box[2]]
    if scale != 1: v = cv2.resize(v, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    p = V / (out or f"{name}_blend{a}{b}.jpg")
    cv2.imwrite(str(p), v, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return p


if __name__ == "__main__":
    print(blend("queens_leonard", 1, 2, (100, 900, 2000, 2000), .9))
    print(blend("queens_leonard", 0, 1, (100, 900, 2000, 2000), .9))
