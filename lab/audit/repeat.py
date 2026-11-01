"""Audit: second independent run of the 45-sheet title read and of F2 door tiles. Writes only to data/lab/audit."""
import sys, json; sys.path.insert(0,"lab/bakeoff"); sys.path.insert(0,"lab/doors"); sys.path.insert(0,"lab")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, fitz
import metalab as m
import e1_meta as e1, truth_meta as tm, sheets as s
from common import rows
A=Path("data/lab/audit"); F=A/"titles_run2.json"
base={x['key']:x for x in rows('meta') if x['cfg']=='s13-low-2k' and x['run']==0}
def one(k):
    r,i=m.spark(e1.PROMPT,images=[s.full(k,2000)],model='muse-spark-1.3',schema=e1.SCHEMA,effort='low')
    return k,r,i
if F.exists(): res=json.load(open(F))
else:
    with ThreadPoolExecutor(6) as ex: res=[[k,r,i] for k,r,i in ex.map(one,list(tm.T))]
    json.dump(res,open(F,"w"),indent=0)
good=lambda q:q['num'] and q['level'] and q['scale']
s2={k:tm.score(k,r) for k,r,_ in res}; s1={k:tm.score(k,base[k]['out']) for k in tm.T}
print('run1',sum(good(v) for v in s1.values()),'run2',sum(good(v) for v in s2.values()),'n',len(s2),'usd',round(sum(i['usd'] for *_,i in res),3))
dev=set(e1.DEV); print('dev n',len(dev&set(tm.T)),'run1 dev',sum(good(s1[k]) for k in dev&set(tm.T)),'run1 fresh',sum(good(s1[k]) for k in set(tm.T)-dev),'/',len(set(tm.T)-dev),'run2 fresh',sum(good(s2[k]) for k in set(tm.T)-dev))
print('misses run2',[(k,{a:b for a,b in v.items() if not b},[x for x in res if x[0]==k][0][1]['sheet_number'],[x for x in res if x[0]==k][0][1]['level'],[x for x in res if x[0]==k][0][1]['scale']) for k,v in s2.items() if not good(v)])
print('flip',[k for k in tm.T if good(s1[k])!=good(s2[k])])
print('spent',round(m.spent('audit'),3))
