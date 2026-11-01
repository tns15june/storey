import sys, json
sys.path.insert(0, 'lab/defects')
from common import *
raw = json.load(open(OUT/'seeds_raw.json'))
tiles=[]
for b,l in [('bme_f','0'),('habs_stelizabeths_center','1'),('habs_va_bldg1','0'),('schenley','0'),('mml_capitol','main')]:
    im=cv2.imread(str(OUT/'stack'/b/f'{l}.png')); h,w=im.shape[:2]
    for k in ('exit_doors','corridors'):
      for i,o in enumerate(raw[b][l][k]):
        x1,y1,x2,y2=o['x_min']*w//1000,o['y_min']*h//1000,o['x_max']*w//1000,o['y_max']*h//1000
        if k=='corridors':
            cx,cy=(x1+x2)//2,(y1+y2)//2; x1,y1,x2,y2=cx-60,cy-60,cx+60,cy+60
        x1,y1,x2,y2=max(x1-50,0),max(y1-50,0),min(x2+50,w),min(y2+50,h)
        c=im[y1:y2,x1:x2]; s=260/max(c.shape[:2]); c=cv2.resize(c,None,fx=s,fy=s)
        c=cv2.copyMakeBorder(c,0,290-c.shape[0],0,264-c.shape[1],cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(c,f'{b[:8]} {l} {k[:2]}{i} {x1},{y1},{x2},{y2}',(1,284),0,.36,(0,0,255),1,cv2.LINE_AA); tiles.append(c)
n=7; rows=[np.hstack((tiles[i:i+n]+[np.full_like(tiles[0],255)]*n)[:n]) for i in range(0,len(tiles),n)]
cv2.imwrite(str(OUT/'thumbs'/'exits.jpg'),np.vstack(rows)); print(len(tiles))
