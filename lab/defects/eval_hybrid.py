"""Hybrid rule over the replayed runs. Fresh runs (id ending _f2 or _f2b...) are reported separately."""
import sys, json, glob
sys.path.insert(0, 'lab/defects')
import numpy as np
from collections import defaultdict
from rules import hybrid, findings, base_find
from pathlib import Path
OUT = Path('data/lab/defects')
SHAFT = {'elevator', 'xshaft_w', 'xshaft_e', 'xshaft_c', 'elev1', 'elev2', 'vent_w', 'vent_e'}
DEV = ('bme_f', 'habs_stelizabeths_center')
R = [json.load(open(p)) for p in glob.glob(str(OUT / 'runs2/*.json')) if not p.endswith('_t.json')]
fn = lambda r: hybrid(r['res'], r['metrics'], r['names'])
sam = lambda r: findings(r['res'], r['metrics'], r['names'])
ink = lambda r: base_find(r['res'], r['metrics'], r['names'], 'ink_seed', .45)
isjit = lambda r: r['case'].get('jitter') is not None
isfresh = lambda r: '_f2' in r['case']['id']
main = [r for r in R if not isjit(r)]
old = [r for r in R if isjit(r) and not isfresh(r)]
new = [r for r in R if isfresh(r)]


def lab(c):
    k = c.get('label') or c['type']
    return f"erase_{c['pos']}" if k == 'erase' else k


def summary(rs, f, name):
    o = {}
    for k in ('erase_interior', 'erase_terminal', 'move', 'move_mid', 'move_30', 'shift_small', 'step', 'wall', 'narrow', 'exit', 'swap_dup', 'swap_order'):
        for g in ('dev', 'hold'):
            s = [r for r in rs if lab(r['case']) == k and ((r['case']['b'] in DEV) == (g == 'dev'))]
            if not s:
                continue
            hit = 0
            for r in s:
                c = r['case']; F = f(r)
                if k in ('shift_small', 'step'):
                    hit += any(x['obj'] == c['obj'] for x in F)
                elif k in ('wall', 'narrow'):
                    hit += any(x['obj'] == 'corridor' and x['floor'] == c['floor'] for x in F)
                elif k == 'exit':
                    hit += any(x['obj'] == 'door' and x['floor'] == c['floor'] for x in F)
                elif k.startswith('swap'):
                    hit += any(x['floor'] in (r['names'][c['slot']], r['names'][c['src']]) for x in F)
                else:
                    hit += any(x['obj'] == c['obj'] and x['floor'] == c['floor'] for x in F)
            o[f'{k}.{g}'] = f'{hit}/{len(s)}'
    cl = defaultdict(list)
    for r in rs:
        if r['case']['type'] == 'clean':
            cl[r['case']['b']].append(len(f(r)))
    # false findings on untouched objects in single-object defect runs, worst building
    ot = defaultdict(list)
    for r in rs:
        c = r['case']
        if c['type'] in ('erase', 'move'):
            ot[c['b']].append(len([x for x in f(r) if x['obj'] != c['obj']]))
    print(name, json.dumps(o))
    print('   clean findings', dict(cl))
    print('   untouched-object findings mean/max', {b[:8]: (round(float(np.mean(v)), 2), max(v)) for b, v in ot.items()})
    return o, dict(cl), {b: (float(np.mean(v)), max(v)) for b, v in ot.items()}


def jit(rs, name):
    for nm, f in (('hybrid', fn), ('SAM rule', sam), ('ink fixed box', ink)):
        hit = tot = fa = 0
        byb = defaultdict(int); per = defaultdict(lambda: [0, 0])
        for r in rs:
            F = f(r); c = r['case']
            if c['type'] == 'erase':
                tot += 1; h = any(x['obj'] == c['obj'] and x['floor'] == c['floor'] for x in F); hit += h
                per[c['b'][:8]][0] += h; per[c['b'][:8]][1] += 1
                wrong = [x for x in F if x['obj'] != c['obj']]
            else:
                wrong = F
            fa += len(wrong); byb[c['b'][:8]] += len(wrong)
        print(name, nm, f'recall {hit}/{tot} false findings {fa} in {len(rs)} runs', dict(byb), dict(per))


if __name__ == '__main__':
    print(len(main), len(old), len(new))
    S = summary(main, fn, 'HYBRID aligned')
    print('misses', [r['case']['id'] for r in main if lab(r['case']) == 'erase_interior' and not any(x['obj'] == r['case']['obj'] and x['floor'] == r['case']['floor'] for x in fn(r))])
    print('false', [(r['case']['id'], [(x['type'], x['obj'], x['floor']) for x in fn(r) if x['obj'] != r['case'].get('obj')]) for r in main if [x for x in fn(r) if x['obj'] != r['case'].get('obj')]][:12])
    print('shift alarms', [(r['case']['id'], [(x['type'], x['obj'], x['floor']) for x in fn(r)]) for r in main if lab(r['case']) in ('shift_small', 'step') and any(x['obj'] == r['case']['obj'] for x in fn(r))][:10])
    stairs = [r for r in main if lab(r['case']) == 'erase_interior']
    for st, nm in ((False, 'stairs'), (True, 'shafts')):
        for g in ('dev', 'hold'):
            s = [r for r in stairs if (r['case']['obj'] in SHAFT) == st and ((r['case']['b'] in DEV) == (g == 'dev'))]
            print(nm, g, sum(any(x['obj'] == r['case']['obj'] and x['floor'] == r['case']['floor'] for x in fn(r)) for r in s), '/', len(s))
    jit(old, 'JITTER first set (rule designed after seeing it)')
    if new:
        jit(new, 'JITTER fresh seeds, frozen rule')
