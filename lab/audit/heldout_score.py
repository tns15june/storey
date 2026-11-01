"""Audit: score the held-out stacks. No API."""
import sys; sys.path.insert(0,"lab/seed")
from common import *
A=ROOT/"data/lab/audit"; T=json.load(open(ROOT/"lab/audit/truth.json")); NF={"habs_va_bldg1":5,"bme_k":4,"wfu_huffman":4}
tot=dict(n=0,seeded=0,best=0,worst=0,anyc=0,spark=0,base=0,floors=0)
pp=dict(present=0,sam_p=0,spk_p=0,flr_p=0,absent=0,sam_a=0,spk_a=0,flr_a=0)
def wh(b):
    im=cv2.imread(str(A/b/"f0.png")); h,w=im.shape[:2]; return w-w%2,h-h%2
for b in NF:
    t=T[b]; nf=NF[b]; w,h=wh(b)
    cl=json.load(open(A/f"{b}_clusters.json")); tr=json.load(open(A/"sam"/f"{b}__union.json"))["tracks"]
    sp={f:[to_px(o,w,h) for o in json.load(open(A/"spark"/f"{b}_f{f}.json"))["objects"]] for f in range(nf)}
    fl=json.load(open(A/"spark"/f"{b}_floors0.json"))["out"]["stairs"]
    claimed=set(); print("==",b,"clusters",len(cl),"spark boxes per floor",[len(v) for v in sp.values()],"floor-compare stairs",len(fl))
    for k,s in t.items():
        bx={int(f):v for f,v in s["boxes"].items()}; any_box=next(iter(bx.values()))
        cands=[i for i,c in enumerate(cl) if any(f in bx and hit(bb,bx[f],.25) for f,bb in c["members"])]
        claimed.update(cands)
        def ev(i,conf):
            o=tr.get(str(i+1),{}); r={}
            for f in range(nf):
                mk=o.get(str(f)); has=bool(mk and mk["area"]>0 and (mk["conf"] or 0)>=conf)
                r[f]=(has and hit(mk["box"],bx[f],.25)) if f in bx else (not has)
            return r
        res=[ev(i,.5) for i in cands]; ok=[all(r.values()) for r in res]
        anyc=any(all(ev(i,0.0001).values()) for i in cands)
        spk={f:(any(hit(bb,bx[f],.25) for bb in sp[f]) if f in bx else not any(hit(bb,any_box,.25) for bb in sp[f])) for f in range(nf)}
        # floor compare: match listed stair by centre distance on floors where truth has it
        best=None
        for st in fl:
            if len(st["floors"])!=nf: continue
            d=[np.hypot(st["floors"][f]["x"]*w/1000-(bx[f][0]+bx[f][2])/2,st["floors"][f]["y"]*h/1000-(bx[f][1]+bx[f][3])/2) for f in bx if st["floors"][f]["present"]]
            if d and (best is None or np.median(d)<best[0]): best=(float(np.median(d)),st)
        fc={f:None for f in range(nf)}
        if best and best[0]<120:
            for f in range(nf): fc[f]=bool(best[1]["floors"][f]["present"])==(f in bx)
        ms=[]
        for f in range(nf):
            row=dict(f=f,truth=f in bx,spark=spk[f],fc=fc[f],sam=[bool(r[f]) for r in res],conf=[(tr.get(str(i+1),{}).get(str(f)) or {}).get("conf") for i in cands])
            ms.append(row)
            if s["cls"]=="main":
                g="present" if f in bx else "absent"; sf="_p" if f in bx else "_a"
                pp[g]+=1; pp["spk"+sf]+=spk[f]; pp["sam"+sf]+=bool(res and any(r[f] for r in res)); pp["flr"+sf]+=bool(fc[f])
        print(f" {k:10s} {s['cls']:4s} seeded {len(cands)} e2e_best {any(ok)} e2e_allclusters {bool(ok) and all(ok)} anyconf {anyc} spark {all(spk.values())} fc_dist {None if not best else round(best[0])}")
        for r in ms:
            if not(r['spark'] and r['fc'] and r['sam'] and all(r['sam'])): print("     ",r)
        if s["cls"]=="main":
            tot["n"]+=1; tot["seeded"]+=bool(cands); tot["best"]+=any(ok); tot["worst"]+=bool(ok) and all(ok); tot["anyc"]+=anyc; tot["spark"]+=all(spk.values()); tot["base"]+=len(bx)==nf; tot["floors"]+=all(v is True for v in fc.values())
    false=[i for i in range(len(cl)) if i not in claimed]
    print(" false clusters",len(false),[(cl[i]["seed_floor"],cl[i]["seed_box"],sorted((tr.get(str(i+1)) or {}).keys())) for i in false])
print(tot); print(pp)
print("-- foreign/blank control: masks on the replaced sheet")
for b,slot in (("bme_k",2),("habs_va_bldg1",2)):
    for n in("foreign","foreign2","blank"):
        d=json.load(open(A/"sam"/f"{b}__ctrl_{n}.json")); c=[(o,v[str(slot)]["conf"]) for o,v in d["tracks"].items() if str(slot) in v and v[str(slot)]["area"]>0]
        clean=json.load(open(A/"sam"/f"{b}__union.json"))["tracks"]
        # did the other floors survive? compare presence on other floors to the clean run
        same=tot2=0
        for o,v in d["tracks"].items():
            for f in range(NF[b]):
                if f==slot: continue
                a=str(f) in v and (v[str(f)]["conf"] or 0)>=.5; bb=str(f) in clean.get(o,{}) and (clean[o][str(f)]["conf"] or 0)>=.5
                tot2+=1; same+=a==bb
        print(b,n,"objects",len(d["seeds"]),"masks on bad sheet",len(c),"conf>=.5",sum((x or 0)>=.5 for _,x in c),"max",max([x or 0 for _,x in c]+[0]),"other floors unchanged",same,"/",tot2)
