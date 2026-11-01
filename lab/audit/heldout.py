"""Audit: recommended seed pipeline on 3 stacks no seed tuning touched, plus Spark one-call floor compare and foreign-sheet controls. Cached per call."""
import sys; sys.path.insert(0,"lab/seed"); sys.path.insert(0,"lab")
from common import *
from concurrent.futures import ThreadPoolExecutor
import metalab as m
A=ROOT/"data/lab/audit"; REP=4
BOX={"type":"object","additionalProperties":False,"required":["object_name","x_min","y_min","x_max","y_max"],"properties":{"object_name":{"type":"string"},**{k:{"type":"integer"} for k in("x_min","y_min","x_max","y_max")}}}
SCHEMA={"type":"object","additionalProperties":False,"required":["objects"],"properties":{"objects":{"type":"array","items":BOX}}}
TAIL=' For each object, answer in the format {"object_name": "<name>", "x_min": <int>, "y_min": <int>, "x_max": <int>, "y_max": <int>}. The coordinates should be in the 0-1000 range, x from the left edge and y from the top edge of the image. If you cannot find an object, omit it.'
DOC='You are an object grounding expert. Provide the bounding box coordinates of the objects in the image: "staircase". Ensure the objects accurately match the request and do not miss any objects.'+TAIL
FLOOR_P="""These images are floor plans of one building, one image per floor, in this order: {order}. Staircases in the same place on consecutive floors are the same staircase.
List every distinct staircase in the building. For each give a short name, and for every floor in the list say whether it is drawn on that floor (present) and its centre x, y in the 0-1000 range of that floor's image (0, 0 if absent). A stair drawn struck through or filled in counts as absent. Compare the floors carefully, stair by stair."""
def obj(**k): return {"type":"object","additionalProperties":False,"required":list(k),"properties":k}
def arr(i): return {"type":"array","items":i}
STR={"type":"string"};INT={"type":"integer"}
FLOOR_S=obj(stairs=arr(obj(name=STR,floors=arr(obj(floor=STR,present={"type":"boolean"},x=INT,y=INT)))))
NF={"habs_va_bldg1":5,"bme_k":4,"wfu_huffman":4}
ORDER={"habs_va_bldg1":"ground, first, second, third, fourth","bme_k":"lower ground, ground, first, second","wfu_huffman":"ground, first, second, third"}
def spark_one(a):
    bid,i=a; p=A/"spark"/f"{bid}_f{i}.json"
    if p.exists(): return json.load(open(p))
    r,info=m.spark(DOC,images=[str(A/bid/f"f{i}.png")],schema=SCHEMA)
    o={"objects":r["objects"],"info":info}; json.dump(o,open(p,"w"),indent=1); print(bid,i,len(r["objects"]),round(info["usd"],4),flush=True); return o
def floors_call(a):
    bid,run=a; p=A/"spark"/f"{bid}_floors{run}.json"
    if p.exists(): return json.load(open(p))
    ims=[str(A/bid/f"f{i}.png") for i in range(NF[bid])]
    r,info=m.spark(FLOOR_P.format(order=ORDER[bid]),images=ims,schema=FLOOR_S,effort="medium",max_tokens=20000)
    o={"out":r,"info":info}; json.dump(o,open(p,"w"),indent=1); print(bid,"floors",run,round(info["usd"],4),flush=True); return o
def linked(a,b):
    area=lambda x:max(1,(x[2]-x[0])*(x[3]-x[1])); cx=lambda x:((x[0]+x[2])/2,(x[1]+x[3])/2)
    s,bg=(a,b) if area(a)<area(b) else (b,a); c=cx(s)
    return iou(a,b)>=.15 or (bg[0]<=c[0]<=bg[2] and bg[1]<=c[1]<=bg[3] and area(bg)<5*area(s))
def cluster(items):
    par=list(range(len(items)))
    def find(i):
        while par[i]!=i: i=par[i]
        return i
    for i in range(len(items)):
        for j in range(i):
            if linked(items[i][1],items[j][1]): par[find(i)]=find(j)
    g={}
    for i in range(len(items)): g.setdefault(find(i),[]).append(items[i])
    out=[]
    for v in g.values():
        med=max(v,key=lambda x:sum(iou(x[1],y[1]) for y in v))
        out.append({"seed_floor":med[0],"seed_box":med[1],"members":v,"floors":sorted({f for f,_ in v})})
    return sorted(out,key=lambda c:-len(c["floors"]))
def token(b,wh): return f"<|box;x1={b[0]};y1={b[1]};x2={b[2]};y2={b[3]};w={wh[0]};h={wh[1]}|>"
def track(tag,frames,seeds):
    p=A/"sam"/f"{tag}.json"
    if p.exists(): return json.load(open(p))
    mp4=A/"sam"/f"{tag}.mp4"; wh=stack_mp4(frames,mp4,REP); n=len(frames)*REP; tr={}
    for s in range(0,len(seeds),16):
        part=seeds[s:s+16]; byf={}
        for o,f,b in part: byf.setdefault(f,[]).append(f"{o}{token(b,wh)}")
        prompt="".join(f"<{f*REP}f>"+",".join(v) for f,v in sorted(byf.items()))
        for r in m.sam_video(prompt,mp4,frames=n):
            fl,sub=r.frame.frame_index//REP,r.frame.frame_index%REP
            if sub==REP-1:
                b=r.bounds; tr.setdefault(str(r.object_id),{})[str(fl)]={"box":[b.left,b.top,b.right,b.bottom],"area":int(m.raster(r).sum()),"conf":None if r.confidence is None else round(r.confidence,3)}
    o={"seeds":[[a,f,list(map(int,b))] for a,f,b in seeds],"tracks":tr}; json.dump(o,open(p,"w")); print(tag,len(seeds),"tracked",len(tr),flush=True); return o
def main():
    (A/"spark").mkdir(exist_ok=True);(A/"sam").mkdir(exist_ok=True)
    jobs=[(b,i) for b in NF for i in range(NF[b])]
    with ThreadPoolExecutor(5) as ex: sp=dict(zip(jobs,ex.map(spark_one,jobs)))
    with ThreadPoolExecutor(3) as ex: list(ex.map(floors_call,[(b,0) for b in NF]))
    for b in NF:
        fr=[cv2.imread(str(A/b/f"f{i}.png")) for i in range(NF[b])]
        h,w=fr[0].shape[:2]; W2,H2=w-w%2,h-h%2
        items=[(f,to_px(o,W2,H2)) for f in range(NF[b]) for o in sp[(b,f)]["objects"]]
        cl=cluster(items); json.dump(cl,open(A/f"{b}_clusters.json","w"),default=int)
        track(f"{b}__union",fr,[(i+1,c["seed_floor"],c["seed_box"]) for i,c in enumerate(cl)])
    # control: one sheet replaced by another building's sheet, or a blank page. Any mask at conf>=.5 there is a false presence.
    for b,slot in (("bme_k",2),("habs_va_bldg1",2)):
        fr=[cv2.imread(str(A/b/f"f{i}.png")) for i in range(NF[b])]; h,w=fr[0].shape[:2]
        cl=json.load(open(A/f"{b}_clusters.json"))
        seeds=[(i+1,c["seed_floor"],c["seed_box"]) for i,c in enumerate(cl) if c["seed_floor"]!=slot][:16]
        for name,src in (("foreign",str(OUT/"schenley/f1.png")),("foreign2",str(OUT/"habs_stelizabeths_center/f1.png")),("blank",None)):
            g=list(fr); g[slot]=np.full_like(fr[0],235) if src is None else cv2.resize(cv2.imread(src),(w,h))
            track(f"{b}__ctrl_{name}",g,seeds)
    print("audit spent",round(m.spent("audit"),3))
main()
