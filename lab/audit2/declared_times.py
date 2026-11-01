"""When did declared's Spark reads happen, by set, against its freeze at 16:13:32. spark.jsonl is append-order, so row i pairs with the i-th 'declared' ledger row. No API."""
import json, collections, datetime as dt
L=[json.loads(x) for x in open('data/lab/ledger.jsonl') if x.strip()]
L=[r for r in L if r['name']=='declared']
S=[json.loads(x) for x in open('data/lab/declared/spark.jsonl') if x.strip()]
print(len(L),len(S))
pref=collections.Counter(r['key'].split('/')[0]+'/'+r['key'].split('/')[1] if '/' in r['key'] else r['key'] for r in S)
print(pref.most_common(40))
first={}
for i,r in enumerate(S):
    k=r['key'].split('/')[1] if '/' in r['key'] else r['key']
    first.setdefault(k,i)
for k,i in sorted(first.items(), key=lambda kv: kv[1]):
    t=L[min(i,len(L)-1)]['t']
    print(k, 'row',i, dt.datetime.fromtimestamp(t).strftime('%H:%M:%S'))
