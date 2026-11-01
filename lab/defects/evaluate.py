"""Score every stored run. DEV buildings fit thresholds, the other three are held out. Writes results.json, prints tables."""
import sys, json, glob
from collections import defaultdict
sys.path.insert(0, 'lab/defects')
import numpy as np
from rules import findings, base_find, P
from pathlib import Path
OUT = Path('data/lab/defects')
DEV = ('bme_f', 'habs_stelizabeths_center')
SHAFT = {'elevator', 'xshaft_w', 'xshaft_e', 'xshaft_c', 'elev1', 'elev2', 'vent_w', 'vent_e'}
runs = [json.load(open(p)) for p in glob.glob(str(OUT / 'runs/*.json')) if not p.endswith('_t.json') and '/jit_' not in p]


def label(c):
    k = c.get('label') or c['type']
    if k == 'erase':
        k = f"erase_{c['pos']}"
    return k


def score(fn, P_=None):
    """fn(run) -> findings. Returns rows of (case, hit, n other-object false findings)."""
    rows = []
    for r in runs:
        c = r['case']; f = fn(r)
        tgt = c.get('obj')
        k = label(c)
        fl = c.get('floor')
        if k in ('erase_interior', 'erase_terminal', 'move', 'move_mid', 'move_30'):
            hit = any(x['obj'] == tgt and x['floor'] == fl for x in f)
            other = [x for x in f if x['obj'] != tgt]
        elif k in ('shift_small', 'step'):
            hit = any(x['obj'] == tgt for x in f)
            other = [x for x in f if x['obj'] != tgt]
        elif k in ('wall', 'narrow'):
            hit = any(x['obj'] == 'corridor' and x['floor'] == fl for x in f)
            other = [x for x in f if x['obj'] != 'corridor']
        elif k == 'exit':
            hit = any(x['obj'] == 'door' and x['floor'] == fl for x in f)
            other = [x for x in f if x['obj'] != 'door']
        elif k in ('swap_dup', 'swap_order'):
            slots = [r['names'][c['slot']], r['names'][c['src']]]
            hit = any(x['floor'] in slots for x in f)
            other = [x for x in f if x['floor'] not in slots]
        else:
            hit = False; other = f
        rows.append(dict(id=c['id'], b=c['b'], k=k, obj=tgt, floor=fl, hit=hit, other=len(other), shaft=tgt in SHAFT if tgt else None, f=f))
    return rows


def table(rows, keys, split=True):
    res = {}
    for k in keys:
        for grp, bs in (('dev', DEV), ('hold', None)):
            s = [r for r in rows if r['k'] == k and ((r['b'] in DEV) == (grp == 'dev'))]
            res[(k, grp)] = (sum(r['hit'] for r in s), len(s))
    return res


def show(name, rows):
    keys = ['erase_interior', 'erase_terminal', 'move', 'move_mid', 'move_30', 'shift_small', 'step', 'wall', 'narrow', 'exit', 'swap_dup', 'swap_order']
    t = table(rows, keys)
    print(f'\n== {name}')
    for k in keys:
        d, h = t[(k, 'dev')], t[(k, 'hold')]
        print(f'  {k:16s} dev {d[0]:3d}/{d[1]:3d}  hold {h[0]:3d}/{h[1]:3d}')
    cl = [r for r in rows if r['k'] == 'clean']
    per = defaultdict(list)
    for r in cl:
        per[r['b']].append(r['other'])
    print('  clean findings per stack:', {b[:10]: v for b, v in per.items()})
    nt = [r for r in rows if r['k'] in ('erase_interior', 'erase_terminal', 'move', 'wall', 'narrow', 'exit', 'shift_small')]
    per = defaultdict(list)
    for r in nt:
        per[r['b']].append(r['other'])
    print('  untouched-object findings per run:', {b[:10]: (round(np.mean(v), 2), max(v)) for b, v in per.items()})
    return t


def jitter_table():
    js = [json.load(open(p)) for p in glob.glob(str(OUT / 'runs/jit_*.json'))]
    out = {}
    for nm, fn in (('SAM rule', lambda r: findings(r['res'], r['metrics'], r['names'])), ('NCC 0.4', lambda r: base_find(r['res'], r['metrics'], r['names'], 'ncc', .4)), ('ink box 0.45', lambda r: base_find(r['res'], r['metrics'], r['names'], 'ink_seed', .45))):
        hit = tot = fa = 0
        for r in js:
            f, c = fn(r), r['case']
            if c['type'] == 'erase':
                tot += 1; hit += any(x['obj'] == c['obj'] and x['floor'] == c['floor'] for x in f)
                fa += len([x for x in f if x['obj'] != c['obj']])
            else:
                fa += len(f)
        out[nm] = (hit, tot, fa, len(js))
    return out


def swaps():
    ss = [r for r in runs if r['case']['type'].startswith('swap') or r['case']['type'] == 'clean']
    res = []
    for r in ss:
        c = r['case']; s = np.array(r['sim']); n = len(s)
        best = max((s[i, j] for i in range(n) for j in range(i)), default=0)
        res.append((c['id'], c['type'], c['b'], round(float(best), 3)))
    return res


if __name__ == '__main__':
    print('runs', len(runs))
    a = score(lambda r: findings(r['res'], r['metrics'], r['names']))
    t = show('SAM tracks + rule', a)
    ex = [(r['id'], r['f']) for r in a if r['k'] == 'clean' and r['f']]
    print('  clean finding examples:', ex[:8])
    print('  interior misses:', [r['id'] for r in a if r['k'] == 'erase_interior' and not r['hit']])
    print('  move misses:', [r['id'] for r in a if r['k'] == 'move' and not r['hit']])
    print('  shift false alarms:', [r['id'] for r in a if r['k'] in ('shift_small', 'step') and r['hit']])
    for st, nm in ((False, 'stairs'), (True, 'shafts')):
        for grp in ('dev', 'hold'):
            s = [r for r in a if r['k'] == 'erase_interior' and r['shaft'] == st and ((r['b'] in DEV) == (grp == 'dev'))]
            print(f'  interior erase {nm} {grp}: {sum(r["hit"] for r in s)}/{len(s)}')
    pb = defaultdict(list)
    for r in a:
        if r['k'] == 'erase_interior':
            pb[r['b']].append(r['hit'])
    print('  interior recall per building:', {b: f'{sum(v)}/{len(v)}' for b, v in pb.items()})
    if len(sys.argv) > 1:
        for th in (.3, .4, .5, .6):
            show(f'baseline NCC thr {th}', score(lambda r: base_find(r['res'], r['metrics'], r['names'], 'ncc', th)))
        for th in (.3, .45, .6):
            show(f'baseline ink in fixed box thr {th}', score(lambda r: base_find(r['res'], r['metrics'], r['names'], 'ink_seed', th)))
