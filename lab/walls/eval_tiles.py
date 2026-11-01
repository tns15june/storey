"""Spark door boxes on tiles: recall against GT doors, then carve into blockers and score. Same 22 test plans. No API calls."""
import json, os, sys
sys.path.insert(0, 'lab/walls')
import multiprocessing as mp
import cv2, numpy as np
import cc, plan as P, score as S
T = 'data/lab/walls/tiles/'


def one(h):
    d = cc.load(h); g = S.gt_setup(d); u = g['u']
    b = json.load(open(T + h.replace('/', '_') + '.json'))['boxes']
    ctr = [S.door_axes(p)[0] for p in d['doors']]
    pad = 0.25 * u
    hit = [any(x[0] - pad <= c[0] <= x[2] + pad and x[1] - pad <= c[1] <= x[3] + pad for x in b) for c in ctr]
    prec = [any(x[0] - pad <= c[0] <= x[2] + pad and x[1] - pad <= c[1] <= x[3] + pad for c in ctr) for x in b]
    out = {}
    blk = P.blockers(d['img'], u)
    heavy = cv2.morphologyEx(P.ink_mask(d['img'], 170), cv2.MORPH_OPEN, P.SQ(max(3, 0.09 * u))) > 0
    for name, w, pd, keep in (('thick', P.walls(d['img'], u, dbl_k=0.08), pad, None), ('block', blk, pad, None), ('block_pad0', blk, 0, None), ('block_keepheavy', blk, 0.1 * u, heavy), ('block_pad0_keepheavy', blk, 0, heavy)):
        cut = np.zeros(w.shape, np.uint8)
        for x in b:
            cv2.rectangle(cut, (int(x[0] - pd), int(x[1] - pd)), (int(x[2] + pd), int(x[3] + pd)), 1, -1)
        w2 = w & ~(cut > 0)
        if keep is not None:
            w2 |= keep & (cut > 0)  # a door box only clears thin ink: heavy wall ink inside it stays
        c = P.close_doors(w, u)
        fp = P.footprint(c, u)
        r = S.score(d, g, w, P.rooms(c, fp, u), P.walkable(w2, fp, u))
        r['house'] = h
        out[name] = r
    out['hit'] = hit; out['prec'] = prec
    return out


if __name__ == '__main__':
    hs = sorted(json.load(open(T + f))['house'] for f in os.listdir(T))
    with mp.Pool(8) as p:
        rs = p.map(one, hs)
    print('plans', len(rs), 'door box recall', round(float(np.mean([x for r in rs for x in r['hit']])), 3), 'of', sum(len(r['hit']) for r in rs), 'precision', round(float(np.mean([x for r in rs for x in r['prec']])), 3), 'of', sum(len(r['prec']) for r in rs))
    tab = {k: S.summary([r[k] for r in rs]) for k in ('thick', 'block', 'block_pad0', 'block_keepheavy', 'block_pad0_keepheavy')}
    for k in tab:
        print(k, json.dumps(tab[k]))
    json.dump(dict(recall=float(np.mean([x for r in rs for x in r['hit']])), precision=float(np.mean([x for r in rs for x in r['prec']])), n=len(rs), tab=tab), open('data/lab/walls/tiles22.json', 'w'), indent=1)
