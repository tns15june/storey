"""What if door locations were perfect? Carve GT door polygons into each blocking mask. No API calls. usage: oracle.py split n"""
import json, sys
sys.path.insert(0, 'lab/walls')
import multiprocessing as mp
import cv2, numpy as np
import cc, plan as P, score as S


def one(h):
    d = cc.load(h); g = S.gt_setup(d); u = g['u']
    if u <= 0 or not d['rooms']:
        return None
    k = cv2.dilate(d['door'].astype(np.uint8), P.SQ(0.5 * u)) > 0
    out = {}
    for name, w in (('thick', P.walls(d['img'], u, dbl_k=0.08)), ('block', P.blockers(d['img'], u))):
        w2 = w & ~k
        c = P.close_doors(w | d['door'], u)
        fp = P.footprint(c, u)
        out[name] = S.score(d, g, w, P.rooms(c, fp, u), P.walkable(w2, fp, u))
        out[name]['house'] = h
    return out


if __name__ == '__main__':
    name, n = sys.argv[1], int(sys.argv[2])
    hs = cc.split(name); rng = np.random.default_rng(0); hs = [hs[i] for i in rng.permutation(len(hs))[:n]]
    with mp.Pool(12) as p:
        rs = [r for r in p.map(one, hs, chunksize=2) if r]
    for k in rs[0]:
        print(k, json.dumps(S.summary([r[k] for r in rs])))
        for st in ('high_quality_architectural', 'high_quality/', 'colorful'):
            print('  ', st, json.dumps(S.summary([r[k] for r in rs if st in r[k]['house']]))[:230])
