"""Per-floor crops of a stair with truth box (green), P track box (red), Ppin box (blue) and conf. No API. Run with -I.
Usage: ts_overlay.py out.jpg stack stair [stack stair ...]"""
import json, sys
import cv2, numpy as np
sys.path.insert(0, "lab/audit2")
import rescore_tightseed as S
R = S.R
def tile(st, k, f, base):
    im = cv2.imread(str(R / "stacks" / st / f"f{f}.png"))
    x1, y1, x2, y2 = base
    pad = 90
    X1, Y1, X2, Y2 = max(int(x1 - pad), 0), max(int(y1 - pad), 0), int(x2 + pad), int(y2 + pad)
    c = im[Y1:Y2, X1:X2].copy()
    z = 300 / max(c.shape[:2])
    c = cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_CUBIC)
    def rect(b, col):
        cv2.rectangle(c, (int((b[0] - X1) * z), int((b[1] - Y1) * z)), (int((b[2] - X1) * z), int((b[3] - Y1) * z)), col, 1)
    s = S.ALL[st]["stairs"][k]
    t = s["boxes"].get(str(f))
    if t: rect(t, (0, 170, 0))
    lab = f"{k} f{f} {'present' if t else 'ABSENT'}"
    y = 12
    for me, col in (("P_960", (0, 0, 230)), ("Ppin_960", (230, 0, 0))):
        tr = S.load(st, me); own = S.owners(st, tr).get(k, ())
        for o in own:
            mk = tr["tracks"].get(o, {}).get(str(f))
            if mk and mk["area"] > 0:
                rect(mk["box"], col)
                y += 12
                cv2.putText(c, f"{me[:4]} {mk['conf']}", (3, y), cv2.FONT_HERSHEY_SIMPLEX, .38, col, 1, cv2.LINE_AA)
    c = cv2.copyMakeBorder(c, 0, 300 - c.shape[0], 0, 300 - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(c, lab, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 120, 0), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(c, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
args = sys.argv[1:]
out, pairs = args[0], list(zip(args[1::2], args[2::2]))
rows = []
for st, k in pairs:
    s = S.ALL[st]["stairs"][k]
    b = next(iter(s["boxes"].values()))
    rows.append(np.hstack([tile(st, k, f, b) for f in range(S.ALL[st]["n"])] + [np.full((302, 302, 3), 255, np.uint8)] * (5 - S.ALL[st]["n"])))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
