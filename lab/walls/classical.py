"""Run classical method over a split in parallel. usage: classical.py split n [json param overrides]"""
import json, sys, time
sys.path.insert(0, 'lab/walls')
import multiprocessing as mp
import numpy as np
import cc, plan as P, score as S


def one(a):
    h, kw = a
    try:
        d = cc.load(h)
        g = S.gt_setup(d)
        if g['u'] <= 0 or not d['rooms']:
            return None
        u = g['u']  # shortcut: scale taken from GT door width; the product reads it from the sheet
        w = P.blockers(d['img'], u, **kw.get('block', {})) if 'block' in kw else P.walls(d['img'], u, **kw.get('walls', {}))
        ds = P.find_doors(d['img'], u, wall=(P.cv2.dilate(P.ink_mask(d['img'], 200), P.SQ(3)) > 0 if kw.get('use_doors') == 2 else w) , **kw.get('doors', {})) if kw.get('use_doors') else []
        w0 = w
        if kw.get('join'):
            cand = P.find_doors(d['img'], u, **kw.get('loose', dict(ring_max=0.6, fill_max=0.3)))
            fp0 = P.footprint(P.close_doors(w, u), u, **kw.get('fp', {}))
            ds = P.join_rooms(w, fp0, u, cand)
        w, bar = P.carve(w, ds, u)
        c = P.close_doors(w | bar, u, **kw.get('close', {}))
        fp = P.footprint(c, u, **kw.get('fp', {}))
        lab = P.rooms(c, fp, u)
        wk = P.walkable(w, fp, u, **kw.get('walk', {}))
        r = S.score(d, g, w0, lab, wk)
        r['doors_found'] = len(ds); r['doors_gt'] = len(d['doors'])
        r['door_hit'] = [bool(ds) and min(float(np.hypot(*((q[0] + q[1]) / 2 - c0))) for q in ds) < 0.7 * u for c0 in [S.door_axes(p)[0] for p in d['doors']]]
        r['door_fp'] = 0 if not ds else sum(min(float(np.hypot(*((q[0] + q[1]) / 2 - S.door_axes(p)[0]))) for p in d['doors']) >= 0.7 * u for q in ds)
        r['house'] = h
        return r
    except Exception as e:
        return dict(err=repr(e), house=h)


if __name__ == '__main__':
    name, n = sys.argv[1], int(sys.argv[2])
    kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    hs = cc.split(name)
    rng = np.random.default_rng(0); hs = [hs[i] for i in rng.permutation(len(hs))[:n]]
    t = time.time()
    with mp.Pool(12) as p:
        res = p.map(one, [(h, kw) for h in hs], chunksize=2)
    errs = [r for r in res if r and 'err' in r]
    res = [r for r in res if r and 'err' not in r]
    print(kw, 'errs', len(errs), errs[:2], 'sec', round(time.time() - t))
    print(json.dumps(S.summary(res)))
    print('door symbols: recall', round(float(np.mean([x for r in res for x in r['door_hit']])), 3), 'found', sum(r['doors_found'] for r in res), 'gt', sum(r['doors_gt'] for r in res), 'fp', sum(r['door_fp'] for r in res))
    for st in ('high_quality_architectural', 'high_quality/', 'colorful'):
        rr = [r for r in res if st in r['house']]
        if rr: print(st, json.dumps(S.summary(rr)))
    if len(sys.argv) > 4:
        json.dump(res, open(sys.argv[4], 'w'))
