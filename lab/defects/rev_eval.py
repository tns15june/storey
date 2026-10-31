"""Score the revision test. Thresholds chosen on dev buildings, applied to holdout."""
import json, numpy as np, sys
r = json.load(open('/Users/tsharma/Github/Meta/data/lab/defects/revision.json'))
DEV = ('bme_f', 'habs_stelizabeths_center')
rows = []
for t, it in r.items():
    for k, v in it.items():
        rows.append(dict(b=t[4:-2], item=k, truth=v['truth'], diff=v['diff'], ink=v.get('ink_ratio'), ncc=v.get('ncc'), mv=v.get('move'), c2=(v['c2'] or {}).get('conf'), none=v['c2'] is None))
def sam(x, mv=12., lo=.6, hi=1.3, conf=.3, ncc=None):
    if x['none'] or x['c2'] < conf: return True
    if ncc is not None and x['ncc'] < ncc: return True
    return x['mv'] > mv or x['ink'] < lo or x['ink'] > hi
def pix(x, t): return x['diff'] > t
def rep(fn, name, rows):
    out = {}
    for g, sel in (('dev', [x for x in rows if x['b'] in DEV]), ('hold', [x for x in rows if x['b'] not in DEV])):
        c = [x for x in sel if x['truth']]; u = [x for x in sel if not x['truth']]
        by = {t: f"{sum(fn(x) for x in c if x['truth'] == t)}/{sum(1 for x in c if x['truth'] == t)}" for t in ('erased', 'moved', 'walled', 'exit inked')}
        out[g] = dict(changed=f'{sum(fn(x) for x in c)}/{len(c)}', by=by, false=f'{sum(fn(x) for x in u)}/{len(u)}')
    print(name, out); return out
res = {}
res['SAM tracks, mv>12 or ink outside .6-1.3'] = rep(sam, 'SAM', rows)
res['SAM tracks + NCC<.6'] = rep(lambda x: sam(x, ncc=.6), 'SAM+ncc.6', rows)
d = [x for x in rows if x['b'] in DEV]
best = max(np.arange(.1, .8, .01), key=lambda t: sum(pix(x, t) for x in d if x['truth']) - sum(pix(x, t) for x in d if not x['truth']))
print('pixel diff thr fitted on dev', round(float(best), 2))
res[f'registered pixel diff > {best:.2f}'] = rep(lambda x: pix(x, best), 'pix', rows)
json.dump(res, open('/Users/tsharma/Github/Meta/data/lab/defects/revision_scores.json', 'w'), indent=1)
