"""Audit: re-score seed e2e with stricter credit. No API calls."""
import sys; sys.path.insert(0,"/Users/tsharma/Github/Meta/lab/seed")
from common import *
from union import cluster, CONF
ALL=["bme_f","schenley","habs_stelizabeths_atkins","habs_stelizabeths_center","mml_capitol","nibs_clinic","mml_baxter"]
tot=dict(n=0,best=0,worst=0,tight=0,tight_worst=0,multi=0,sam_any=0)
def ev(t,tr,nf,conf,thr=.25,box_of=None):
    res=[]
    for f in range(nf):
        if f in t.get("any",[]): continue
        mk=tr.get(str(f)); has=bool(mk and mk.get("area",0)>0 and (mk.get("conf") or 0)>=conf)
        if f in t["floors"]:
            res.append(has and hit(mk["box"],t["box"],thr))
        else: res.append(not has)
    return res
det=[]
for bid in ALL:
    nf=len(CAT[bid]["floors"]); cl=cluster(bid,"doc")
    tr=json.load(open(OUT/"sam"/f"union_doc__{bid}.json"))["tracks"]
    for t in TRUTH[bid]:
        if t["cls"]!="main": continue
        c=[i for i,x in enumerate(cl) if any(f in t["floors"] and hit(b,t["box"],.25) for f,b in x["members"])]
        tot["n"]+=1
        # how many clusters seed box itself (medoid) hits truth
        seedhit=[i for i in c if hit(cl[i]["seed_box"],t["box"],.25)]
        scores=[(i,all(ev(t,tr.get(str(i+1),{}),nf,CONF))) for i in c]
        best=any(s for _,s in scores); worst=all(s for _,s in scores)
        tot["best"]+=best; tot["worst"]+=worst
        tot["multi"]+=len(c)>1
        an=any(all(ev(t,tr.get(str(i+1),{}),nf,0.0)) for i in c); tot["sam_any"]+=an
        det.append((bid,t["id"],len(c),len(seedhit),[(i,s) for i,s in scores],t.get("any")))
print(tot)
for d in det: print(d)
# truth box sizes relative to per-floor Spark boxes
