"""Overlay of the soft-plant floor as actually sent, with the returned mask box and conf. No API."""
import json, sys
import cv2, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/audit2")
import sam_erasure as E
core = E.core
tiles = []
for st, picks in (("bme_k", ("N_center", "W_grand", "E_wing", "spiral_NE")), ("habs_va_bldg1", ("west", "central"))):
    fr = E.frames(st, "soft")[E.M]
    clean = core.frames(st)[E.M]
    tr = json.load(open(core.SAMD / f"{st}__erase_soft.json"))
    ks = E.stairs(st)
    for k in picks:
        i = ks.index(k) + 1
        mk = tr["tracks"][str(i)][str(E.M)]
        b = core.TRUTH[st]["stairs"][k]["boxes"][str(E.M)]
        pad = 70
        X1, Y1, X2, Y2 = max(int(b[0]) - pad, 0), max(int(b[1]) - pad, 0), int(b[2]) + pad, int(b[3]) + pad
        ca = clean[Y1:Y2, X1:X2].copy(); cb = fr[Y1:Y2, X1:X2].copy()
        for c in (ca, cb):
            cv2.rectangle(c, (int(b[0]) - X1, int(b[1]) - Y1), (int(b[2]) - X1, int(b[3]) - Y1), (0, 170, 0), 1)
        cv2.rectangle(cb, (int(mk["box"][0]) - X1, int(mk["box"][1]) - Y1), (int(mk["box"][2]) - X1, int(mk["box"][3]) - Y1), (0, 0, 230), 2)
        q = np.hstack([ca, np.full((ca.shape[0], 6, 3), 0, np.uint8), cb])
        z = 480 / q.shape[1]
        q = cv2.resize(q, None, fx=z, fy=z, interpolation=cv2.INTER_AREA)
        q = cv2.copyMakeBorder(q, 16, 300 - min(q.shape[0], 284) - 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))[:300]
        cv2.putText(q, f"{st[:8]} {k} left clean, right as sent, conf {mk['conf']}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, .38, (200, 0, 0), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(q, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
cv2.imwrite(str(E.AUD / "sam_soft_view.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
