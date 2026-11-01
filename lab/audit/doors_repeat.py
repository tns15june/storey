"""Audit: second run of F2 5x5 low-effort door tiles against the IFC key. Writes only to data/lab/audit."""
import sys, json; sys.path.insert(0,"lab"); sys.path.insert(0,"lab/doors")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, fitz
import metalab as m, key as K, spark_tiles as T
F=Path("data/lab/audit/doors_F2_run2.json")
tag="F2"; A,L,arcs,reg=K.clinic_key(tag); G=[T.arc_point(l) for l in L]; h=np.array(G)
box=(h[:,0].min()-30,h[:,1].min()-30,h[:,0].max()+30,h[:,1].max()+30)
def one(rc):
    dpi=int(round(72*1600/max(rc[2]-rc[0],rc[3]-rc[1]))); im=K.render(tag,dpi,clip=fitz.Rect(*rc))
    r,info=m.spark(T.PROMPT,images=[im],schema=T.SCHEMA,effort="low")
    return {"rect":list(rc),"doors":r["points"],"info":info}
if F.exists(): outs=json.load(open(F))
else:
    with ThreadPoolExecutor(6) as ex: outs=list(ex.map(one,T.tiles(box,5)))
    json.dump(outs,open(F,"w"))
pts=T.nms([p for o in outs for p in T.to_sheet(o)],.45*K.PT_PER_M)
for tol in (.5,.8):
    tp=T.score(pts,G,tol*K.PT_PER_M); print(tol,'recall',round(tp/len(G),3),'precision',round(tp/len(pts),3),'pts',len(pts))
print('usd',round(sum(o['info']['usd'] for o in outs),3),'spent',round(m.spent('audit'),3))
print('original',json.load(open(K.OUT/'spark_tiles_F2.json'))['t5low'])
