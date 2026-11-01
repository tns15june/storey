"""Audit: fairest case for SAM. Tight hand boxes as prompts instead of Spark boxes, same stacks, same scoring."""
import sys; sys.path.insert(0,"lab/seed"); sys.path.insert(0,"lab")
from common import *
import metalab as m
src=open(ROOT/"lab/audit/heldout.py").read().replace("\nmain()\n","\n"); exec(src)
A=ROOT/"data/lab/audit"; T=json.load(open(ROOT/"lab/audit/truth.json"))
res={}
for b,fp in (("bme_k",0),("bme_k",1),("habs_va_bldg1",1)):
    t={k:v for k,v in T[b].items() if v["cls"]=="main" and str(fp) in v["boxes"]}
    names=list(t); fr=[cv2.imread(str(A/b/f"f{i}.png")) for i in range(NF[b])]
    d=track(f"{b}__hand_f{fp}",fr,[(i+1,fp,t[k]["boxes"][str(fp)]) for i,k in enumerate(names)])
    print("==",b,"prompt floor",fp)
    for i,k in enumerate(names):
        bx={int(f):v for f,v in t[k]["boxes"].items()}; o=d["tracks"].get(str(i+1),{}); row=[]
        for f in range(NF[b]):
            mk=o.get(str(f)); c=mk and mk["conf"]; has=bool(mk and mk["area"]>0 and (c or 0)>=.5)
            okk=(has and hit(mk["box"],bx[f],.25)) if f in bx else not has
            row.append(f"f{f}:{'P' if f in bx else 'A'}:{c}:{'ok' if okk else 'BAD'}")
            res[(b,fp,k,f)]=(f in bx,okk,c)
        print(f" {k:10s}",*row)
for fp_b in (("bme_k",0),("bme_k",1),("habs_va_bldg1",1)):
    b,fp=fp_b
    r=[v for kk,v in res.items() if kk[0]==b and kk[1]==fp and kk[3]!=fp]
    print(b,fp,"unprompted present",sum(x[1] for x in r if x[0]),"/",sum(1 for x in r if x[0]),"absent",sum(x[1] for x in r if not x[0]),"/",sum(1 for x in r if not x[0]))
print('spent',round(m.spent('audit'),3))
