"""Classical walkable mask on scanned and raster sheets. u is a door width in pixels, set by eye from the sheet. usage: scans.py"""
import sys, json
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/walls')
import cv2, fitz, numpy as np, plan as P, vec
D = '/Users/tsharma/Github/Meta/data/plans/'; O = '/Users/tsharma/Github/Meta/data/lab/walls/'
def load(k):
    if k == 'af_ces_admin':
        return vec.render(fitz.open(D + 'af_ces_admin/CES_SD_drawings_admin_bldg.pdf')[1], 150)
    return cv2.imread(D + k)
SHEETS = {'schenley': ('schenley/sch_first.png', 26), 'bme_f': ('bme_f/bme_0.jpg', 22), 'af_ces_admin': ('af_ces_admin', 45), 'bme_k': ('bme_k/02.jpg', 55)}
for name, (k, u) in SHEETS.items():
    im = load(k)
    s = min(1.0, 3000 / max(im.shape[:2]))
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    u2 = u
    print(name, im.shape, 'u', u2)
    w = P.blockers(im, u2, floor=150)
    w = cv2.morphologyEx(w.astype(np.uint8), cv2.MORPH_CLOSE, P.SQ(3)) > 0
    arcs = P.find_doors(im, u2, wall=w)
    w2, bar = P.carve(w, arcs, u2)
    c = P.close_doors(w2 | bar, u2)
    fp = P.footprint(c, u2, fp_gap=4)
    wk = P.walkable(w2, fp, u2)
    lab = P.rooms(c, fp, u2)
    print(' arcs', len(arcs), 'rooms', len(np.unique(lab)) - 1, 'walk frac', round(float(wk.mean()), 3))
    n, cc_, st, _ = cv2.connectedComponentsWithStats(wk.astype(np.uint8), connectivity=8)
    big = np.argsort(-st[1:, cv2.CC_STAT_AREA])[:6] + 1
    v = (im * .45 + 140).astype(np.uint8)
    cols = [(90, 200, 90), (80, 80, 230), (230, 160, 60), (60, 200, 230), (200, 90, 200), (120, 120, 120)]
    for i, b in enumerate(big):
        v[cc_ == b] = (v[cc_ == b] * .45 + np.array(cols[i]) * .55).astype(np.uint8)
    v[w] = (0, 0, 0)
    for H, E, L, sc in arcs:
        cv2.line(v, tuple(H.astype(int).tolist()), tuple(E.astype(int).tolist()), (255, 0, 255), 3)
    np.savez_compressed(O + f'scan_{name}.npz', walk=wk, wall=w)
    z = 1500 / max(v.shape[:2])
    cv2.imwrite(O + f'scan_{name}.jpg', cv2.resize(v, None, fx=z, fy=z, interpolation=cv2.INTER_AREA))
    print(' components top', [int(st[b, cv2.CC_STAT_AREA]) for b in big])
