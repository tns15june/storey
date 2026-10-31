"""Audit: no-model baseline for seed e2e: claim every main stair exists at one place on every floor."""
import json
t=json.load(open('/Users/tsharma/Github/Meta/lab/seed/truth.json'))
cat={x['id']:x for x in json.load(open('/Users/tsharma/Github/Meta/data/plans/catalog.json'))}
tot=ok=0;ends=[]
for b,v in t.items():
    if b.startswith('_'):continue
    nf=len(cat[b]['floors'])
    for s in v:
        if s['cls']!='main':continue
        tot+=1
        eff=[f for f in range(nf) if f not in s.get('any',[])]
        missing=[f for f in eff if f not in s['floors']]
        if missing: ends.append((b,s['id'],missing))
        else: ok+=1
print('always-present baseline',ok,'/',tot,'stacks with a real absent floor',ends)
