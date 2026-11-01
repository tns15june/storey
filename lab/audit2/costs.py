import json, collections
rows=[json.loads(x) for x in open('data/lab/ledger.jsonl') if x.strip()]
for name in ('tightseed','presence','agent','samtricks'):
    c=collections.defaultdict(list)
    for r in rows:
        if r['name']==name:
            c[(r['model'], r.get('kind'))].append(r['usd'])
    print(name, {str(k):(len(v), round(sum(v),3), round(sum(v)/len(v),4)) for k,v in c.items()})
# presence big calls
p=[r for r in rows if r['name']=='presence' and r.get('images') and r['images']>=3]
print('presence multi-image', len(p), 'mean', round(sum(r['usd'] for r in p)/len(p),4), 'max', max(r['usd'] for r in p))
p=[r for r in rows if r['name']=='presence' and r.get('images')==1 and r['usd']<0.02]
print('presence single image', len(p), 'mean', round(sum(r['usd'] for r in p)/len(p),5))
print(rows[-1])
print([r for r in rows if r['name']=='tightseed' and 'sam' in r['model'].lower()][:2])
