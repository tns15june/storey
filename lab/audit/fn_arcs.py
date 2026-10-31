"""Audit: for door leaves the detector missed, is there a drawn arc nearby that matching rejected? And do Spark 'false' points sit on those same doors? No API."""
import sys, json; sys.path.insert(0,'/Users/tsharma/Github/Meta/lab/doors')
import numpy as np, key as K
from vector_eval import midpoint, match_arcs, TOL_MID, footprint, in_box, quarter
for tag in K.SHEETS:
    A,L,arcs,reg=K.clinic_key(tag)
    leafy=[a for a in arcs if a["leaf"]]
    box=footprint(L); pool=[a for a in arcs if in_box(a,box) and (a["leaf"] or quarter(a))]
    m=match_arcs(pool,L,TOL_MID); hit={j for _,j in m}; used={i for i,_ in m}
    fn=[j for j in range(len(L)) if j not in hit]
    G=np.array([midpoint(l["s_hinge"],l["s_shut"]) for l in L])
    C=np.array([[a["cx"],a["cy"]] for a in arcs])
    M=np.array([midpoint([a["cx"],a["cy"]],a["a"]) for a in arcs]+[midpoint([a["cx"],a["cy"]],a["b"]) for a in arcs])
    rows=[]
    for j in fn:
        dc=np.linalg.norm(C-G[j],axis=1); dm=np.linalg.norm(M-G[j],axis=1)
        rows.append((L[j]["name"][2:14],round(float(dc.min()/K.PT_PER_M),2),round(float(dm.min()/K.PT_PER_M),2)))
    print(tag,'FN',len(fn),'nearest arc centre/chord-mid m',rows)
    print(' FN with any arc centre <1.2m:',sum(r[1]<1.2 for r in rows),' tol m',round(TOL_MID/K.PT_PER_M,2))
