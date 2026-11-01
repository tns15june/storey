"""Align each building's floors to its reference floor, crop to the plan, write stack/<b>/<level>.png, report the ECC score."""
import sys
sys.path.insert(0, 'lab/defects')
from common import *
for b in (sys.argv[1:] or BUILDINGS):
    fl = CAT[b]['floors'][::-1]  # top floor first
    sheets = [load_sheet(b, f, LONG.get(b, 3000)) for f in fl]
    r = len(fl) - 1 - REF[b]
    ref = sheets[r]
    al = []
    for i, s in enumerate(sheets):
        if i == r: al.append(ref); continue
        w, A, cc, inl = align(s, ref, WALLK.get(b, 5), ROI.get(b), fixed=b == 'mml_capitol')
        print(b, fl[i]['level'], 'ecc', round(cc, 3), 'inliers', inl, flush=True)
        al.append(w)
    h, w0 = ref.shape[:2]
    if b in CROP:
        x1, y1, x2, y2 = CROP[b]; bx = (int(x1 * w0), int(y1 * h), int(x2 * w0), int(y2 * h))
    else:
        g = ink(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)) > 40
        g = cv2.morphologyEx(np.uint8(g), cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(g)
        keep = [i for i in range(1, n) if st[i, 4] > 0.002 * h * w0 and st[i, 0] > 0.01 * w0 and st[i, 1] > .01 * h and st[i, 0] + st[i, 2] < .99 * w0 and st[i, 1] + st[i, 3] < .99 * h]
        xs = [st[i, 0] for i in keep] + [st[i, 0] + st[i, 2] for i in keep]; ys = [st[i, 1] for i in keep] + [st[i, 1] + st[i, 3] for i in keep]
        m = 60; bx = (max(min(xs) - m, 0), max(min(ys) - m, 0), min(max(xs) + m, w0), min(max(ys) + m, h))
    d = OUT / 'stack' / b; d.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = bx
    s = W / max(x2 - x1, y2 - y1)
    out = []
    for f, im in zip(fl, al):
        c = cv2.resize(im[y1:y2, x1:x2], None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        c = c[:c.shape[0] - c.shape[0] % 2, :c.shape[1] - c.shape[1] % 2]; cv2.imwrite(str(d / f"{f['level']}.png"), c); out.append(c)
    print(b, 'crop', bx, 'size', out[0].shape, flush=True)
    # red/cyan blend of each floor against the reference to judge alignment
    g = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in out]
    rows = [cv2.resize(cv2.merge([g[r], g[r], g[i]]), None, fx=.5, fy=.5) for i in range(len(g)) if i != r]
    cv2.imwrite(str(OUT / 'thumbs' / f'align_{b}.jpg'), np.vstack(rows))
