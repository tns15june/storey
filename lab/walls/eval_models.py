"""All methods on the same 22 CubiCasa test plans. Uses cached Spark and SAM results only, no API calls."""
import json, os, sys
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/walls')
import multiprocessing as mp
import cv2, numpy as np
import cc, plan as P, score as S
SP = '/Users/tsharma/Github/Meta/data/lab/walls/spark/'; SM = '/Users/tsharma/Github/Meta/data/lab/walls/sam/'
KW = dict(dbl_k=0.08)


def sp_points(sp, W, H):
    return [(np.array([p['point'][0] * W / 1000, p['point'][1] * H / 1000]), p['kind']) for p in sp['doors'] if len(p['point']) == 2]


def discs(shape, pts, r):
    m = np.zeros(shape, np.uint8)
    for p in pts:
        cv2.circle(m, (int(p[0]), int(p[1])), int(r), 1, -1)
    return m > 0


def run_classical(d, u, w, ds):
    w2, bar = P.carve(w, ds, u)
    c = P.close_doors(w2 | bar, u)
    fp = P.footprint(c, u)
    return P.rooms(c, fp, u), P.walkable(w2, fp, u), fp


def one(h):
    d = cc.load(h); g = S.gt_setup(d); u = g['u']; H, W = d['h'], d['w']
    sp = json.load(open(SP + h.replace('/', '_') + '.json')); sm = np.load(SM + h.replace('/', '_') + '.npz')
    pts = [p for p, k in sp_points(sp, W, H)]
    w = P.walls(d['img'], u, **KW)
    out = {}
    # classical alone
    lab, wk, fp = run_classical(d, u, w, [])
    out['a_classical'] = S.score(d, g, w, lab, wk)
    ring = P.find_doors(d['img'], u, wall=w)
    lab, wk, fp = run_classical(d, u, w, ring)
    out['a_classical_arcs'] = S.score(d, g, w, lab, wk)
    # candidates with loose filters, accepted only near a Spark door point (mix d)
    loose = P.find_doors(d['img'], u, ring_max=0.6, fill_max=0.3, wall=None)
    acc = [q for q in loose if pts and min(np.hypot(*((q[0] + q[1]) / 2 - p)) for p in pts) < 1.0 * u]
    lab, wk, fp = run_classical(d, u, w, acc)
    out['d_classical_arcs_sparkgate'] = S.score(d, g, w, lab, wk)
    out['d_classical_arcs_sparkgate']['cand'] = (len(loose), len(acc))
    # plus: Spark points with no accepted arc get a disc carved through the wall
    un = [p for p in pts if not acc or min(np.hypot(*((q[0] + q[1]) / 2 - p)) for q in acc) > 1.0 * u]
    w3 = w & ~discs(w.shape, un, 0.55 * u)
    lab2, wk2, fp2 = run_classical(d, u, w3, acc)
    out['d_arcs_sparkgate_discs'] = S.score(d, g, w, lab2, wk2)
    # SAM rooms from Spark boxes: walkable is rooms plus door discs at Spark points, walls the complement
    for name in ('raw', 'clean'):
        L = sm[name]
        rooms = L > 0
        walk = (rooms | discs(rooms.shape, pts, 0.55 * u))
        pw = cv2.morphologyEx(rooms.astype(np.uint8), cv2.MORPH_CLOSE, P.SQ(0.6 * u)) > 0
        pw = pw & ~rooms
        out['b_spark_sam_' + name] = S.score(d, g, pw, L, walk)
        # hybrid: SAM room masks with classical walls subtracted, classical door carving
        wk3 = (walk & ~w) | (discs(rooms.shape, pts, 0.55 * u) & rooms)
        out['d_sam_%s_minus_walls' % name] = S.score(d, g, w, L, wk3 > 0)
    for v in out.values():
        v['house'] = h
    return out


if __name__ == '__main__':
    hs = sorted(json.load(open(SP + f))['house'] for f in os.listdir(SP))
    hs = [h for h in hs if os.path.exists(SM + h.replace('/', '_') + '.npz')]
    with mp.Pool(8) as p:
        rs = p.map(one, hs)
    tab = {}
    for k in rs[0]:
        tab[k] = S.summary([r[k] for r in rs])
        print(k, json.dumps(tab[k]))
    print('cand', [r['d_classical_arcs_sparkgate']['cand'] for r in rs][:8])
    json.dump(tab, open('/Users/tsharma/Github/Meta/data/lab/walls/models22.json', 'w'), indent=1)
    json.dump(rs, open('/Users/tsharma/Github/Meta/data/lab/walls/models22_plans.json', 'w'))
