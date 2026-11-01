"""Shared loading, alignment and stack encoding for the planted defect suite."""
import json, subprocess, sys
from pathlib import Path
import cv2, numpy as np, fitz

ROOT = Path('.')
PLANS = ROOT / 'data/plans'
OUT = ROOT / 'data/lab/defects'
W = 2000
CAT = {e['id']: e for e in json.load(open(PLANS / 'catalog.json'))}
# reference floor, top-to-bottom floor order is the stack order
BUILDINGS = ['bme_f', 'habs_stelizabeths_center', 'habs_va_bldg1', 'schenley', 'mml_capitol']
REF = {'bme_f': 1, 'habs_stelizabeths_center': 1, 'habs_va_bldg1': 1, 'schenley': 1, 'mml_capitol': 1}
# plan-area crop as fractions of the reference sheet, so SAM spends pixels on the plan
CROP = {'mml_capitol': (0.57, 0.04, 0.80, 0.76), 'habs_va_bldg1': (0.12, 0.13, 0.92, 0.72), 'schenley': (0.0, 0.0, 1.0, 0.88)}
LONG = {'mml_capitol': 6000}
ROI = {'habs_va_bldg1': (0.17, 0.27, 0.86, 0.60), 'mml_capitol': (0.59, 0.20, 0.78, 0.62)}
WALLK = {'schenley': 3, 'mml_capitol': 1}


def load_sheet(b, f, long=3000):
    p = PLANS / b / f['file']
    if p.suffix.lower() == '.pdf':
        pg = fitz.open(p)[f['page']]
        z = long / max(pg.rect.width, pg.rect.height)
        px = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
        return np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, 3)[..., ::-1].copy()
    im = cv2.imread(str(p))
    s = long / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def ink(gray):
    """Ink as a positive signal, flat paper removed."""
    bg = cv2.medianBlur(cv2.resize(gray, None, fx=.25, fy=.25), 31)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]))
    return np.clip(bg.astype(np.float32) - gray, 0, 255)


def walls(gray, k=5):
    """Thick strokes only: walls survive between floors, text and fixtures do not."""
    I = ink(gray) > 45
    o = cv2.morphologyEx(np.uint8(I), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return np.float32(o)


def align(mov, ref, wall_k=5, roi=None, fixed=False):
    """Mov onto ref: normalised template match of the reference plan region over scale and rotation on blurred wall maps, then ECC kept only if it stays near."""
    wm, wr = walls(cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY), wall_k), walls(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), wall_k)
    f = 600 / max(ref.shape[:2])
    a0 = cv2.GaussianBlur(cv2.resize(wm, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    b0 = cv2.GaussianBlur(cv2.resize(wr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    H, W_ = b0.shape
    x1, y1, x2, y2 = roi or (0.1, 0.1, 0.9, 0.9)
    tx, ty = int(x1 * W_), int(y1 * H)
    tpl = b0[ty:int(y2 * H), tx:int(x2 * W_)]
    best = (-1, None, 0, 0)
    pad = 60
    for sc in ([1.0] if fixed else np.arange(0.90, 1.1001, 0.004)):
        for rot in ([0.0] if fixed else np.arange(-3, 3.01, 0.5)):
            M = cv2.getRotationMatrix2D((a0.shape[1] / 2, a0.shape[0] / 2), rot, sc)
            M[:, 2] += pad
            a = cv2.warpAffine(a0, M, (W_ + 2 * pad, H + 2 * pad))
            r = cv2.matchTemplate(a, tpl, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(r)
            if v > best[0]:
                best = (v, np.vstack([M, [0, 0, 1]]), tx - loc[0], ty - loc[1])
    v, M, dx, dy = best
    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float) @ M
    S = np.diag([f, f, 1.0])
    A = np.float32((np.linalg.inv(S) @ T @ S)[:2])
    s2 = 1000 / max(ref.shape[:2])
    a, b = [cv2.GaussianBlur(cv2.resize(x, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA), (0, 0), 1.5) for x in (wm, wr)]
    A2 = A.copy(); A2[:, 2] *= s2
    try:
        cc, A3 = cv2.findTransformECC(a, b, A2, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-6), None, 1)
        A3[:, 2] /= s2
        h, w = ref.shape[:2]
        c = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32).T
        if roi or np.abs(A3 @ c - A @ c).max() > 0.03 * max(w, h):  # with a roi ECC would see the frame lines again
            A3, cc = A, -2
    except cv2.error:
        cc, A3 = -1, A
    bv = tuple(int(x) for x in np.median(ref.reshape(-1, 3), axis=0))
    return cv2.warpAffine(mov, A3, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR, borderValue=bv), A3, float(cc), float(v)


def stack_mp4(frames, path, repeat=4):
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w2}x{h2}', '-r', '4', '-i', '-',
                          '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    for f in frames:
        for _ in range(repeat):
            p.stdin.write(f[:h2, :w2].tobytes())
    p.stdin.close(); p.wait()
    return w2, h2, len(frames) * repeat
