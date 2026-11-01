"""One renderer for every case, good or bad, so style carries no hint. Colours are BGR. No API."""
import sys
sys.path.insert(0, "lab/agent")
import cv2, numpy as np, pymupdf as fitz
MARK, START, END = (255, 0, 255), (0, 160, 0), (255, 0, 0)  # magenta marker, green start, blue end
LONG = 1100


def render(pg, clip, long=LONG, max_dpi=240):
    """Crop of a PDF page in points. Returns BGR image and pt->px mapper. `clip` is in the rotated page frame, the frame of get_pixmap."""
    r = fitz.Rect(clip)
    dpi = min(max_dpi, long * 72 / max(r.width, r.height))
    # Whole page then numpy crop: pixmap clip on a rotated page is a trap, and this frame is the one vec.py and measure.py report in.
    s = int(dpi) / 72
    pm = pg.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
    full = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)
    full = full[:, :, ::-1] if pm.n == 3 else cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = [int(round(v * s)) for v in r]
    x0, y0 = max(x0, 0), max(y0, 0)
    im = full[y0:y1, x0:x1].copy()
    return im, (lambda x, y: (int(round(x * s)) - x0, int(round(y * s)) - y0)), s


def route(im, P, pts):
    q = np.array([P(x, y) for x, y in pts], np.int32)
    cv2.polylines(im, [q], False, MARK, 3, cv2.LINE_AA)
    cv2.circle(im, tuple(int(v) for v in q[0]), 9, START, -1, cv2.LINE_AA)
    cv2.circle(im, tuple(int(v) for v in q[-1]), 9, END, -1, cv2.LINE_AA)
    return im


def box(im, a, b):
    cv2.rectangle(im, a, b, MARK, 3, cv2.LINE_AA)
    return im


def leader(im, a, b, r=24):
    """Circle at the tag, line to the door, ring at the door end."""
    cv2.circle(im, a, r, MARK, 3, cv2.LINE_AA)
    cv2.line(im, a, b, MARK, 3, cv2.LINE_AA)
    cv2.circle(im, b, 11, MARK, 3, cv2.LINE_AA)
    return im


def ticks(im, a, b, L=16):
    """Two short ticks across the span from a to b, and a thin line between them."""
    d = np.array(b, float) - np.array(a, float)
    n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9) * L
    for p in (a, b):
        cv2.line(im, (int(p[0] + n[0]), int(p[1] + n[1])), (int(p[0] - n[0]), int(p[1] - n[1])), MARK, 4, cv2.LINE_AA)
    cv2.line(im, a, b, MARK, 1, cv2.LINE_AA)
    return im
