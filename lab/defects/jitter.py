"""Stress: every non-seed floor shifted by a random offset up to JIT px, as with poorly aligned scans. Same interior erasures, plus clean stacks."""
import sys, random
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/defects')
import suite
from suite import *
JIT = 60
_orig_stack = {b: suite.STACK[b] for b in SEEDS}


def shift(im, dx, dy):
    bv = tuple(int(x) for x in np.median(im.reshape(-1, 3), axis=0))
    return cv2.warpAffine(im, np.float32([[1, 0, dx], [0, 1, dy]]), (im.shape[1], im.shape[0]), borderValue=bv)


def jit_case(c):
    b = c['b']; fr, names = _orig_stack[b]
    rng = random.Random(c['id'])
    si = names.index(SEEDS[b]['floor'])
    off = [(0, 0) if i == si else (rng.randint(-JIT, JIT), rng.randint(-JIT, JIT)) for i in range(len(names))]
    if c['type'] == 'erase':
        # plant first on the aligned sheet, then jitter, so truth stays known
        i = names.index(c['floor'])
        x1, y1, x2, y2 = boxes_on(b, c['obj'], c['floor'])
        pl = [f.copy() for f in fr]
        pl[i] = erase(pl[i], x1 + 2, y1 + 2, x2 - 2, y2 - 2, seed=1)
    else:
        pl = [f.copy() for f in fr]
    pl = [shift(f, *o) for f, o in zip(pl, off)]
    seeds = base_seeds(b); sf = {}
    res_extra = None
    if c.get('floor') == SEEDS[b]['floor'] and c.get('obj') in seeds:
        o = c['obj']; alt = [x for x in TRUTH[b][o] if x != c['floor'] and boxes_on(b, o, x)]
        ai = names.index(min(alt, key=lambda x: abs(names.index(x) - si)))
        bx = boxes_on(b, o, names[ai]); bx = (bx[0] + off[ai][0], bx[1] + off[ai][1], bx[2] + off[ai][0], bx[3] + off[ai][1])
        seeds.pop(o); res_extra = (o, bx, ai)
    tag = 'jit_' + c['id']
    p = RES / f'{tag}.json'
    if p.exists():
        return tag
    res = run(pl, seeds, tag, si, names)
    alls = dict(seeds)
    if res_extra:
        o, bx, ai = res_extra
        res[o] = run(pl, {o: bx}, tag + '_t', ai, names)[o]; alls[o] = bx; sf[o] = ai
    m_ = metrics(pl, names, res, alls, si, sf)
    c2 = dict(c); c2['jitter'] = off
    p.write_text(json.dumps(dict(case=c2, names=names, res=res, metrics=m_, sim=[], seed_floor=SEEDS[b]['floor'], sf=sf)))
    return tag


if __name__ == '__main__':
    all_ = cases(); random.Random(5).shuffle(all_)
    cs, seen = [c for c in all_ if c['id'].endswith('_r1')], {}
    for c in all_:  # shortcut: budget, four interior erasures per building
        if c['type'] == 'erase' and c['pos'] == 'interior' and seen.get(c['b'], 0) < 4:
            cs.append(c); seen[c['b']] = seen.get(c['b'], 0) + 1
    print(len(cs), flush=True)
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(jit_case, cs)):
            if i % 10 == 0:
                print(i, r, round(m.spent('defects'), 3), flush=True)
