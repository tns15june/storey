"""Overlay montage: usage view.py split out.jpg start count [params json]"""
import json, sys
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/walls')
import cv2, numpy as np
import cc, plan as P, score as S
split, out, st, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
hs = cc.split(split); rng = np.random.default_rng(0); hs = [hs[i] for i in rng.permutation(len(hs))]
tiles = []
for h in hs[st:st + n]:
    d = cc.load(h); g = S.gt_setup(d); u = g['u']
    w = P.blockers(d['img'], u, **kw.get('block', {})) if 'block' in kw else P.walls(d['img'], u, **kw.get('walls', {})); c = P.close_doors(w, u, **kw.get('close', {})); fp = P.footprint(c, u, **kw.get('fp', {}))
    lab = P.rooms(c, fp, u); wk = P.walkable(w, fp, u, **kw.get('walk', {}))
    im = d['img'].copy()
    a = (im * .5 + 127).astype(np.uint8); a[d['wall'] & ~w] = (0, 0, 255); a[w & ~d['wall']] = (255, 0, 0); a[w & d['wall']] = (0, 170, 0)
    b = (im * .4 + 150).astype(np.uint8); b[wk] = (b[wk] * .5 + np.array([0, 110, 127])).astype(np.uint8)
    dd = P.geodesic(wk, g['exits']) if g['exits'] else np.full(wk.shape, np.inf)
    for x, y in g['exits']: cv2.circle(b, (int(x), int(y)), 14, (0, 0, 255), 4)
    f = np.isfinite(dd); b[wk & ~f] = (60, 60, 200)
    s = 620 / max(im.shape); print(h, 'u', round(u, 1), im.shape)
    tiles.append(cv2.resize(np.hstack([a, b]), None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
W = max(t.shape[1] for t in tiles)
cv2.imwrite(out, np.vstack([cv2.copyMakeBorder(t, 0, 6, 0, W - t.shape[1], cv2.BORDER_CONSTANT, value=(120, 120, 120)) for t in tiles]))
