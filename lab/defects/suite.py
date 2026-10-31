"""The planted defect suite. Each case edits real sheets, runs the tracker, stores per-floor metrics. Cached: rerun is free.
Usage: suite.py [pilot|all]"""
import sys, random, traceback, time
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/defects')
from track import *
from plant import *
from seeds import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

DEV = ['bme_f', 'habs_stelizabeths_center']
RES = OUT / 'runs'; RES.mkdir(exist_ok=True, parents=True)
STACK = {b: load_stack(b) for b in SEEDS}
CLEAN = {b: json.load(open(CACHE / f'clean_{b}.json')) for b in SEEDS}


def wmap(im):
    return cv2.GaussianBlur(cv2.resize(walls(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 3), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA), (0, 0), 1.5)


def inkmap(im):
    return ink(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))


def metrics(frames, names, res, seed_boxes, seed_idx, sf):
    """No-AI baseline and ink checks, stored with each run. sf: seed floor index per object."""
    inks = [inkmap(f) for f in frames]
    out = {}
    for o, box in seed_boxes.items():
        x1, y1, x2, y2 = box
        tpl = cv2.GaussianBlur(inks[sf.get(o, seed_idx)][y1:y2, x1:x2], (0, 0), 1)
        out[o] = {}
        for i, n in enumerate(names):
            w = 60
            X1, Y1, X2, Y2 = max(x1 - w, 0), max(y1 - w, 0), min(x2 + w, frames[i].shape[1]), min(y2 + w, frames[i].shape[0])
            r = cv2.matchTemplate(cv2.GaussianBlur(inks[i][Y1:Y2, X1:X2], (0, 0), 1), tpl, cv2.TM_CCOEFF_NORMED)
            ncc = float(np.nan_to_num(r).max())
            t = res[o][n]
            bx = t['box'] if t else box
            a, b_, c, d = max(bx[0], 0), max(bx[1], 0), bx[2], bx[3]
            out[o][n] = dict(ncc=round(ncc, 3), ink=round(float((inks[i][b_:d, a:c] > 30).mean()), 4) if d > b_ and c > a else 0.0, ink_seed=round(float((inks[i][y1:y2, x1:x2] > 30).mean()), 4))
    return out


def base_seeds(b):
    return {k: v[1] for k, v in SEEDS[b]['objects'].items()}


def boxes_on(b, o, f):
    r = CLEAN[b][o].get(f)
    return r['box'] if r and (r['conf'] or 0) >= .3 else None


def do(case):
    p = RES / f"{case['id']}.json"
    if p.exists():
        return case['id'], 'cached'
    b = case['b']
    fr, names = STACK[b]
    fr = [f.copy() for f in fr]
    name2i = {n: i for i, n in enumerate(names)}
    k = case['type']
    seeds = base_seeds(b)
    si = name2i[SEEDS[b]['floor']]
    if k in ('erase', 'move'):
        o, f = case['obj'], case['floor']
        x1, y1, x2, y2 = boxes_on(b, o, f)
        s = 2 if k == 'erase' else 0
        if k == 'erase':
            fr[name2i[f]] = erase(fr[name2i[f]], x1 + s, y1 + s, x2 - s, y2 - s, seed=1)
        else:
            for ff in case.get('floors', [f]):
                bb = boxes_on(b, o, ff) or (x1, y1, x2, y2)
                fr[name2i[ff]] = move(fr[name2i[ff]], bb[0], bb[1], bb[2], bb[3], case['dx'], case['dy'], seed=1)
    elif k == 'clean':
        pass
    elif k == 'swap_dup':
        fr[case['slot']] = rescan(fr[case['src']], 3)
    elif k == 'swap_order':
        i, j = case['slot'], case['src']; fr[i], fr[j] = fr[j], fr[i]
    elif k == 'wall':
        f, ax, lo, hi, at, a, c = case['corr']
        i = name2i[f]; fr[i] = draw_wall(fr[i], at, lo, at, hi, 5) if ax == 'h' else draw_wall(fr[i], lo, at, hi, at, 5)
        seeds['corridor'] = (at - 40, lo - 4, at + 40, hi + 4) if ax == 'h' else (lo - 4, at - 40, hi + 4, at + 40)
    elif k == 'narrow':
        f, ax, lo, hi, at, a, c = case['corr']
        i = name2i[f]; mid = (lo + hi) // 2
        fr[i] = draw_wall(fr[i], a, mid, c, mid, 5) if ax == 'h' else draw_wall(fr[i], mid, a, mid, c, 5)
        seeds['corridor'] = (at - 40, lo - 4, at + 40, hi + 4) if ax == 'h' else (lo - 4, at - 40, hi + 4, at + 40)
    elif k == 'exit':
        f, x1, y1, x2, y2, t = case['ex']
        i = name2i[f]; fr[i] = draw_wall(erase(fr[i], x1 - 4, y1 - 20, x2 + 4, y2 + 20, seed=1), x1, y1, x2, y2, t)
        seeds['door'] = (x1 - 12, y1 - 22, x2 + 12, y2 + 22)
    # a defect on the seed floor cannot be prompted there, so that object gets its own request from another floor
    sf = {}
    extra = None
    if case.get('obj') and case.get('floor') == SEEDS[b]['floor'] and case.get('obj') in seeds:
        o = case['obj']
        alt = [x for x in TRUTH[b][o] if x != SEEDS[b]['floor'] and boxes_on(b, o, x)]
        if alt:
            alt = min(alt, key=lambda x: abs(name2i[x] - si))
            extra = (o, boxes_on(b, o, alt), name2i[alt])
            seeds.pop(o)
    res = run(fr, seeds, case['id'], si, names)
    all_seed = dict(seeds)
    if extra:
        o, bx, ai = extra
        res2 = run(fr, {o: tuple(bx)}, case['id'] + '_t', ai, names)
        res[o] = res2[o]; all_seed[o] = tuple(bx); sf[o] = ai
    m_ = metrics(fr, names, res, all_seed, si, sf)
    sim = np.eye(len(fr))
    if k.startswith('swap') or k == 'clean':
        w_ = [wmap(f) for f in fr]
        for i in range(len(fr)):
            for j in range(i):
                sim[i, j] = sim[j, i] = float(cv2.matchTemplate(w_[i], w_[j], cv2.TM_CCOEFF_NORMED).max())
    p.write_text(json.dumps(dict(case=case, names=names, res=res, metrics=m_, sim=sim.tolist(), seed_floor=SEEDS[b]['floor'], sf=sf)))
    if case.get('grid'):
        grid(fr, res, names, all_seed, OUT / 'thumbs' / f"g_{case['id']}.jpg")
    return case['id'], 'ran'


def cases():
    cs = []
    for b in SEEDS:
        fr, names = STACK[b]
        for r in (1, 2):
            cs.append(dict(id=f'clean_{b}_r{r}', b=b, type='clean'))
        for o, fl in TRUTH[b].items():
            int_ = fl[1:-1]
            for f in fl:
                if boxes_on(b, o, f):
                    pos = 'interior' if f in int_ else 'terminal'
                    cs.append(dict(id=f'erase_{b}_{o}_{f}', b=b, type='erase', obj=o, floor=f, pos=pos))
            if not int_:
                continue
            f = int_[len(int_) // 2]
            bx = boxes_on(b, o, f)
            if not bx:
                continue
            w, h = bx[2] - bx[0], bx[3] - bx[1]
            ik = inkmap(fr[names.index(f)])
            # move along the axis whose destination holds the least ink, staying on the sheet
            opts = []
            for ax, sgn in (('x', 1), ('x', -1), ('y', 1), ('y', -1)):
                L = int(1.2 * (w if ax == 'x' else h))
                dx, dy = (sgn * L, 0) if ax == 'x' else (0, sgn * L)
                X1, Y1, X2, Y2 = bx[0] + dx, bx[1] + dy, bx[2] + dx, bx[3] + dy
                if X1 < 0 or Y1 < 0 or Y2 >= ik.shape[0] or X2 >= ik.shape[1]:
                    continue
                opts.append(((ik[Y1:Y2, X1:X2] > 30).mean(), ax, sgn))
            _, ax, sgn = min(opts)
            for lab, frac in (('move', 1.2), ('move_mid', .6), ('move_30', .3), ('shift_small', .15)):
                d = max(int(frac * (w if ax == 'x' else h)), 6)
                dx, dy = (sgn * d, 0) if ax == 'x' else (0, sgn * d)
                cs.append(dict(id=f'{lab}_{b}_{o}_{f}', b=b, type='move', label=lab, obj=o, floor=f, dx=dx, dy=dy))
            d = max(int(.3 * (w if ax == 'x' else h)), 6)
            dx, dy = (sgn * d, 0) if ax == 'x' else (0, sgn * d)
            above = fl[:fl.index(f) + 1]
            cs.append(dict(id=f'step_{b}_{o}_{f}', b=b, type='move', label='step', obj=o, floor=f, floors=above, dx=dx, dy=dy))
        si = names.index(SEEDS[b]['floor'])
        slots = [i for i in range(len(names)) if i != si]
        for s in slots[:3]:
            src = min([i for i in range(len(names)) if i != s], key=lambda i: abs(i - s))
            cs.append(dict(id=f'swapdup_{b}_{s}', b=b, type='swap_dup', slot=s, src=src))
        a, c = slots[0], slots[-1]
        cs.append(dict(id=f'swaporder_{b}_{a}_{c}', b=b, type='swap_order', slot=a, src=c))
        for i, c_ in enumerate(CORRIDORS.get(b, [])):
            cs.append(dict(id=f'wall_{b}_{i}', b=b, type='wall', corr=c_, floor=c_[0]))
            cs.append(dict(id=f'narrow_{b}_{i}', b=b, type='narrow', corr=c_, floor=c_[0]))
        for i, e in enumerate(EXITS.get(b, [])):
            cs.append(dict(id=f'exit_{b}_{i}', b=b, type='exit', ex=e, floor=e[0]))
    return cs


if __name__ == '__main__':
    cs = cases()
    from collections import Counter
    print(len(cs), Counter((c.get('label') or c['type']) + ('_' + c.get('pos') if 'pos' in c else '') for c in cs), flush=True)
    if sys.argv[1] == 'pilot':
        random.Random(3).shuffle(cs); cs = cs[:8]
    # shortcut: budget. move-size sweeps stay on dev buildings, terminal erases on holdout keep one floor per object
    done_ids = {p.stem for p in RES.glob('*.json')}
    seen = set(); kept = []
    for c in cs:
        lab = c.get('label') or c['type']
        if c['b'] not in DEV and lab in ('move_mid', 'move_30'):
            continue
        if c['b'] not in DEV and c.get('pos') == 'terminal':
            if (c['b'], c['obj']) in seen:
                continue
            seen.add((c['b'], c['obj']))
        kept.append(c)
    cs = [c for c in kept if c['id'] not in done_ids]
    print('to run', len(cs), flush=True)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(do, c): c for c in cs}
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
            except m.BudgetExceeded as e:
                print('BUDGET', e, flush=True); ex.shutdown(cancel_futures=True); break
            except Exception as e:
                print('FAIL', futs[f]['id'], repr(e)[:200], flush=True); traceback.print_exc(); continue
            if done % 10 == 0 or sys.argv[1] == 'pilot':
                print(done, r, round(time.time() - t0), 'spent', round(m.spent('defects'), 3), flush=True)
