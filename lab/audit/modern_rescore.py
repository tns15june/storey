"""Audit: recount modern tracker claims and a no-model baseline. No API."""
import json
B='/Users/tsharma/Github/Meta/data/lab/modern/'
T=json.load(open(B+'track.json'));T2=json.load(open(B+'track2.json'));T3=json.load(open(B+'track3.json'))
for name,tt in (('track',T),('track2',T2),('track3',T3)):
    for r in tt:
        rows=r['rows']; pres=[x for x in rows if x['gt']=='present']; abs_=[x for x in rows if x['gt']=='absent']
        print(name,r['tag'],r.get('variant',''),'pairs',r['pairs'],'correct',r['correct'],'present',sum(x['ok'] for x in pres),len(pres),'absent',sum(x['ok'] for x in abs_),len(abs_),'minconf',min([x.get('conf') or 9 for x in rows if x.get('mask')]+[9]))
