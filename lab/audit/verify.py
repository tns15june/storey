"""Audit: view held-out ghosts, one-to-one check, relaxed scoring, 2 more floor-compare runs."""
import sys; sys.path.insert(0,"/Users/tsharma/Github/Meta/lab/seed"); sys.path.insert(0,"/Users/tsharma/Github/Meta/lab")
from common import *
import metalab as m, re
src=open(ROOT/"lab/audit/heldout.py").read().replace("\nmain()\n","\n")
exec(src)
A=ROOT/"data/lab/audit"; T=json.load(open(ROOT/"lab/audit/truth.json"))
with ThreadPoolExecutor(6) as ex: list(ex.map(floors_call,[(b,r) for b in NF for r in (1,2)]))
def wh(b):
    im=cv2.imread(str(A/b/"f0.png")); h,w=im.shape[:2]; return w-w%2,h-h%2
# montages: each row one truth stair, each column one floor; magenta truth or last-known box, green SAM box with conf from every overlapping cluster
rows=[("bme_k","svc_NW"),("bme_k","svc_SW"),("habs_va_bldg1","east"),("habs_va_bldg1","central"),("wfu_huffman","central")]
out=[]
for b,k in rows:
    bx={int(f):v for f,v in T[b][k]["boxes"].items()}; nf=NF[b]
    cl=json.load(open(A/f"{b}_clusters.json")); tr=json.load(open(A/"sam"/f"{b}__union.json"))["tracks"]
    cands=[i for i,c in enumerate(cl) if any(f in bx and hit(bb,bx[f],.25) for f,bb in c["members"])]
    row=[]
    for f in range(nf):
        im=cv2.imread(str(A/b/f"f{f}.png")); t=bx.get(f,next(iter(bx.values())))
        cx,cy=(t[0]+t[2])//2,(t[1]+t[3])//2; R=170
        x0,y0=max(cx-R,0),max(cy-R,0); s=2.0
        c=cv2.resize(im[y0:cy+R,x0:cx+R],None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC)
        P=lambda x,y:(int((x-x0)*s),int((y-y0)*s))
        cv2.rectangle(c,P(t[0],t[1]),P(t[2],t[3]),(255,0,255) if f in bx else (200,200,0),2)
        yy=16
        for i in cands:
            mk=tr.get(str(i+1),{}).get(str(f))
            if mk:
                bb=mk["box"]; cv2.rectangle(c,P(bb[0],bb[1]),P(bb[2],bb[3]),(0,160,0),2)
                cv2.putText(c,f"c{i} {mk['conf']}",P(bb[0],bb[1]-3),0,.45,(0,120,0),1)
        cv2.putText(c,f"{k} f{f} {'PRESENT' if f in bx else 'absent'}",(4,yy),0,.5,(0,0,255),1)
        c=cv2.copyMakeBorder(c,0,max(0,680-c.shape[0]),0,max(0,680-c.shape[1]),cv2.BORDER_CONSTANT,value=(255,255,255))[:680,:680]
        row.append(c)
    while len(row)<5: row.append(np.full((680,680,3),255,np.uint8))
    out.append(np.hstack(row))
cv2.imwrite(str(A/"ghosts.jpg"),cv2.resize(np.vstack(out),None,fx=.6,fy=.6,interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,82])
# floor compare, one to one, every run
tt=dict(pp=0,pn=0,ap=0,an=0,stacks=0,n=0)
for b in NF:
    w,h=wh(b); t={k:v for k,v in T[b].items() if v["cls"]=="main"}; nf=NF[b]
    for r in (0,1,2):
        fl=json.load(open(A/"spark"/f"{b}_floors{r}.json"))["out"]["stairs"]
        pairs=[]
        for k,s in t.items():
            bx={int(f):v for f,v in s["boxes"].items()}
            for i,st in enumerate(fl):
                if len(st["floors"])!=nf: continue
                d=[np.hypot(st["floors"][f]["x"]*w/1000-(bx[f][0]+bx[f][2])/2,st["floors"][f]["y"]*h/1000-(bx[f][1]+bx[f][3])/2) for f in bx if st["floors"][f]["present"]]
                if d: pairs.append((float(np.median(d)),k,i))
        used=set();got={};
        for d,k,i in sorted(pairs):
            if k in got or i in used or d>120: continue
            got[k]=i;used.add(i)
        ok=0;bad=[]
        for k,s in t.items():
            bx={int(f):v for f,v in s["boxes"].items()}
            if k not in got: bad.append((k,"unmatched"));tt["n"]+=1;tt["pn"]+=len(bx);tt["an"]+=nf-len(bx);continue
            st=fl[got[k]];this=True
            for f in range(nf):
                p=bool(st["floors"][f]["present"])
                if f in bx: tt["pn"]+=1;tt["pp"]+=p
                else: tt["an"]+=1;tt["ap"]+=not p
                if p!=(f in bx): this=False;bad.append((k,f,p))
            tt["n"]+=1;tt["stacks"]+=this
        print(b,"run",r,"listed",len(fl),"matched",len(got),"unused",len(fl)-len(used),"bad",bad)
print(tt,'spent',round(m.spent('audit'),3))
# relaxed: a Spark box counts if its centre is within 60 px of the truth centre
tot=dict(pres=0,pok=0,abs=0,aok=0,stacks=0,okst=0,sam_p=0,sam_a=0,sam_as=0,samok=0,samn=0)
for b in NF:
    w,h=wh(b); nf=NF[b]
    sp={f:[to_px(o,w,h) for o in json.load(open(A/"spark"/f"{b}_f{f}.json"))["objects"]] for f in range(nf)}
    cl=json.load(open(A/f"{b}_clusters.json")); tr=json.load(open(A/"sam"/f"{b}__union.json"))["tracks"]
    near=lambda bb,t,R=60:np.hypot((bb[0]+bb[2]-t[0]-t[2])/2,(bb[1]+bb[3]-t[1]-t[3])/2)<R
    for k,s in T[b].items():
        if s["cls"]!="main":continue
        bx={int(f):v for f,v in s["boxes"].items()}; ab=next(iter(bx.values())); all_ok=True
        for f in range(nf):
            if f in bx: ok=any(near(bb,bx[f]) for bb in sp[f]);tot["pres"]+=1;tot["pok"]+=ok
            else: ok=not any(near(bb,ab) for bb in sp[f]);tot["abs"]+=1;tot["aok"]+=ok
            all_ok&=ok
        tot["stacks"]+=1;tot["okst"]+=all_ok
        c=[i for i,x in enumerate(cl) if any(f in bx and near(bb,bx[f]) for f,bb in x["members"])]
        def fok(i,f):
            mk=tr.get(str(i+1),{}).get(str(f)); has=bool(mk and mk["area"]>0 and (mk["conf"] or 0)>=.5)
            return (has and near(mk["box"],bx[f],80)) if f in bx else not has
        tot["samn"]+=1;tot["samok"]+=any(all(fok(i,f) for f in range(nf)) for i in c)
        for f in range(nf):
            if f not in bx and c: tot["sam_as"]+=1;tot["sam_a"]+=any(fok(i,f) for i in c)
print("relaxed",tot)
