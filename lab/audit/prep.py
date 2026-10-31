"""Audit held-out stacks: aligned 2000 px sheets and grids. No API calls."""
import sys; sys.path.insert(0,"/Users/tsharma/Github/Meta/lab/seed")
from common import *
A=ROOT/"data/lab/audit"; A.mkdir(exist_ok=True)
for bid,ref in (("habs_va_bldg1",1),("bme_k",1),("wfu_huffman",1)):
    d=A/bid; d.mkdir(exist_ok=True)
    fl=CAT[bid]["floors"]; raw=[load_floor(bid,f) for f in fl]
    r0=raw[ref]; info={}
    for i,im in enumerate(raw):
        if im.shape!=r0.shape: pass
        al,mm=(r0,{"method":"ref","overlap":1.0}) if i==ref else align(im,r0)
        cv2.imwrite(str(d/f"f{i}.png"),al); info[i]=mm
        g=al.copy(); h,w=g.shape[:2]
        for x in range(0,w,200):
            cv2.line(g,(x,0),(x,h),(0,0,255),1); cv2.putText(g,str(x),(x+3,28),0,.9,(0,0,255),2)
        for y in range(0,h,200):
            cv2.line(g,(0,y),(w,y),(0,0,255),1); cv2.putText(g,str(y),(3,y+28),0,.9,(0,0,255),2)
        cv2.imwrite(str(d/f"g{i}.jpg"),cv2.resize(g,None,fx=.8,fy=.8,interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,80])
        print(bid,i,fl[i]["level"],mm["method"],mm["overlap"],al.shape)
    json.dump(info,open(d/"align.json","w"),indent=1)
