"""Recompute every bakeoff number from saved results. Makes no API call. Writes data/lab/bakeoff/results.json."""
import json
import math
import sys

sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/bakeoff')
import numpy as np
import e1_meta as e1
import e2_tasks as t
import e3_pdf_floors as e3
import truth_meta as tm
from common import rows, s

R = {}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(c - h, 3), round(c + h, 3)


meta = {}
for cfg in e1.CFG:
    for scope, keys in (('dev12', e1.DEV), ('all45', list(tm.T))):
        r = [x for x in rows('meta') if x['cfg'] == cfg and x['key'] in keys and x['run'] == 0]
        if not r or (scope == 'all45' and len(r) < 45):
            continue
        sc = [tm.score(x['key'], x['out']) for x in r]
        all3 = sum(q['num'] and q['level'] and q['scale'] for q in sc)
        meta[f'{cfg}/{scope}'] = dict(n=len(r), cls=sum(q['cls'] for q in sc), num=sum(q['num'] for q in sc), level=sum(q['level'] for q in sc), scale=sum(q['scale'] for q in sc),
                                      all3=all3, ci=wilson(all3, len(r)), usd=round(sum(x['usd'] for x in r) / len(r), 4), sec=round(sum(x['seconds'] for x in r) / len(r), 1),
                                      input=round(sum(x['input'] for x in r) / len(r)), output=round(sum(x['output'] for x in r) / len(r)))
R['meta'] = meta
for k, v in meta.items():
    print(k, v)
print('misses, spark 1.3 low 2000 px, all 45')
kind = lambda k: 'raster' if s.ALL[k][1] is None or k.startswith('af') else 'vector'
split = {}
for x in rows('meta'):
    if x['cfg'] == 's13-low-2k' and x['run'] == 0:
        q = tm.score(x['key'], x['out'])
        good = q['num'] and q['level'] and q['scale']
        split.setdefault(kind(x['key']), [0, 0]); split[kind(x['key'])][0] += good; split[kind(x['key'])][1] += 1
        if not good:
            print(' ', x['key'], q, x['out']['sheet_number'], '|', x['out']['level'], '|', x['out']['scale'], '| conf', x['out']['confidence'])
print('by kind', split); R['meta_by_kind'] = split
c2 = [(x['cfg'], x['out']['level']) for x in rows('meta') if x['key'] == 'c2']
R['c2_level_reads'] = c2
print('c2 title block says MAIN, drawing says GROUND:', sum(tm.norm_level(b) == 'g' for _, b in c2), 'of', len(c2), 'configs answered ground;', [a for a, b in c2 if tm.norm_level(b) != 'g'])

st = {}
for cfg in t.TASKS['stairs'][3]:
    f = w = 0; e = []; iou = []; n = 0
    us = []
    for k in t.TASKS['stairs'][2]:
        r = t.merged('stairs', k, cfg, 0)
        q = t.score_stairs(k, r['items'])
        f += q['found']; w += q['want']; e += q['err']; iou += q['iou']; n += len(r['items']); us.append((r['usd'], r['sec']))
    st[cfg] = dict(found=f, want=w, median_center_err_pct=round(float(np.median(e)), 2), median_iou=round(float(np.median(iou)), 2), predicted=n, usd=round(np.mean([u[0] for u in us]), 4), sec=round(np.mean([u[1] for u in us]), 1))
    print('stairs', cfg, st[cfg])
R['stairs'] = st

for name, keys, cfgs, fn in (('rooms', t.TASKS['rooms'][2], ['s13-med-2k', 's12-med-2k', 's13-med-t2', 's13-med-t3'], lambda k, r: t.score_rooms(k, r['items'])),
                             ('doors', t.TASKS['doors'][2], ['s13-med-2k', 's13-med-strip'], lambda k, r: t.score_doors(k, r['items']))):
    R[name] = {}
    for cfg in cfgs:
        for k in keys:
            r = t.merged(name, k, cfg, 0)
            if r['raw']:
                R[name][f'{cfg}/{k}'] = dict(fn(k, r), usd=round(r['usd'], 4), sec=round(r['sec'], 1))
                print(name, cfg, k, R[name][f'{cfg}/{k}'])

R['exits'] = {}
for cfg in t.TASKS['exits'][3]:
    for k in t.TASKS['exits'][2]:
        r = t.merged('exits', k, cfg, 0)
        q = t.score_exits(k, r)
        tagged = [i for i in r['items'] if i['capacity'] > 0]
        R['exits'][f'{cfg}/{k}'] = dict(found=q['found'], want=q['want'], capacity_right=q['caps'], extra=q['extra'], tagged=len(tagged), tagged_sum=sum(i['capacity'] for i in tagged), dist=q['dist'], median_err_pct=round(float(np.median(q['err'])), 2) if q['err'] else None, usd=round(r['usd'], 4), sec=round(r['sec'], 1))
        print('exits', cfg, k, R['exits'][f'{cfg}/{k}'], [x for z in r['raw'] for x in z['out']['travel_distances']][:5])

R['code'] = {}
for cfg in ('s13-low-2k', 's13-med-2k', 's12-med-2k', 'w-med-2k'):
    for k in t.TASKS['code'][2]:
        x = [r for r in rows('code') if r['cfg'] == cfg and r['key'] == k][0]
        R['code'][f'{cfg}/{k}'] = dict(t.score_code(k, x['out']), read=x['out'], usd=round(x['usd'], 4), sec=round(x['seconds'], 1))
        print('code', cfg, k, x['out'])

R['pdf_single'] = {}
for exp, cfg in (('pdfmeta', 'pdf-low'), ('meta', 's13-low-2k')):
    r = [x for x in rows(exp) if x['cfg'] == cfg and x['key'] in e3.DEV_PDF and x['run'] == 0]
    sc = [tm.score(x['key'], x['out']) for x in r]
    R['pdf_single'][exp] = dict(n=len(r), **{f: sum(q[f] for q in sc) for f in ('cls', 'num', 'level', 'scale')}, usd=round(sum(x['usd'] for x in r) / len(r), 4), sec=round(sum(x['seconds'] for x in r) / len(r), 1), input=round(sum(x['input'] for x in r) / len(r)))
R['pdf_set'] = {f"{r['key']}/{r['cfg']}": dict(right_of_pages=e3.score_set(r, e3.CAP_TRUTH if r['key'] == 'capitol' else e3.USC_TRUTH), usd=round(r['usd'], 4), sec=round(r['seconds'], 1), input=r['input'], output=r['output']) for r in rows('pdfset')}
print(R['pdf_single'], R['pdf_set'])
R['floors'] = [dict(building=r['key'], cfg=r['cfg'], run=r['run'], pairs=e3.score_floors(r['key'], r['out'])[:2], stairs_listed=len(r['out']['stairs']), detail=e3.score_floors(r['key'], r['out'])[3], usd=round(r['usd'], 4), sec=round(r['seconds'], 1)) for r in rows('floors')]
ok = [f for f in R['floors'] if f['cfg'] == 's13-med']
print('spark 1.3 floors pooled', sum(f['pairs'][0] for f in ok), 'of', sum(f['pairs'][1] for f in ok), 'over', len(ok), 'calls')
led = [json.loads(x) for x in open(s.ROOT / 'data/lab/ledger.jsonl')]
b = [x for x in led if x['name'] == 'bakeoff']
R['ledger'] = dict(calls=len(b), usd=round(sum(x['usd'] for x in b), 3), saved_usd=round(sum(x['usd'] for x in rows('meta') + rows('stairs') + rows('rooms') + rows('exits') + rows('code') + rows('doors') + rows('floors') + rows('pdfmeta') + rows('pdfset')), 3),
                   runaway=[dict(model=x['model'], usd=x['usd'], output=x['output']) for x in b if x.get('output', 0) > 100000])
print(R['ledger'])
R['tokens'] = json.load(open(s.OUT / 'tokens.json')); R['cache'] = json.load(open(s.OUT / 'cache.json'))
json.dump(R, open(s.OUT / 'results.json', 'w'), indent=1, default=str)
