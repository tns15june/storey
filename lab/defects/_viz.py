import sys
sys.path.insert(0, 'lab/defects')
from track import *; from plant import *; from seeds import *
tiles=[]
def tile(a,b,box,lab,pad=50):
    x1,y1,x2,y2=box; x1,y1=max(x1-pad,0),max(y1-pad,0); x2+=pad; y2+=pad
    c=np.hstack([a[y1:y2,x1:x2],b[y1:y2,x1:x2]]); s=480/c.shape[1]; c=cv2.resize(c,None,fx=s,fy=s)
    c=cv2.copyMakeBorder(c,0,300-c.shape[0] if c.shape[0]<300 else 0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))[:300]
    cv2.putText(c,lab,(2,14),0,.42,(0,0,255),1,cv2.LINE_AA); return c
for b,o,fl in [('bme_f','II','1'),('habs_stelizabeths_center','west_curve','3'),('habs_va_bldg1','central','2'),('schenley','sw','1'),('mml_capitol','rear','main'),('habs_va_bldg1','elev2','3')]:
    fr,names=load_stack(b); i=names.index(fl); r=json.load(open(OUT/'tracks'/f'clean_{b}.json'))[o][fl]; x1,y1,x2,y2=r['box']
    z=np.load(OUT/'tracks'/f'clean_{b}.npz'); sh=z[f'{o}|{fl}|shape']; mk=np.unpackbits(z[f'{o}|{fl}'])[:sh[0]*sh[1]].reshape(sh).astype(bool)
    reg=None
    e=erase(fr[i],x1,y1,x2,y2,reg); tiles.append(tile(fr[i],e,(x1,y1,x2,y2),f'{b[:6]} {o} erase'))
    m=move(fr[i],x1,y1,x2,y2,int(1.2*(x2-x1)),0,reg); tiles.append(tile(fr[i],m,(x1-20,y1,x2+int(1.4*(x2-x1)),y2),f'{b[:6]} {o} move'))
for b,cs in CORRIDORS.items():
    fr,names=load_stack(b); f,ax,lo,hi,at,a,c=cs[0]; im=fr[names.index(f)]
    w=draw_wall(im,at,lo,at,hi,5) if ax=='h' else draw_wall(im,lo,at,hi,at,5)
    n=draw_wall(im,a,(lo+hi)//2,c,(lo+hi)//2,5) if ax=='h' else draw_wall(im,(lo+hi)//2,a,(lo+hi)//2,c,5)
    bx=(at-80,lo-40,at+80,hi+40) if ax=='h' else (lo-40,at-80,hi+40,at+80)
    tiles.append(tile(w,n,bx,f'{b[:6]} wall|narrow'))
for b,es in EXITS.items():
    fr,names=load_stack(b); f,x1,y1,x2,y2,t=es[0]; im=fr[names.index(f)]
    e=draw_wall(erase(im,x1-4,y1-20,x2+4,y2+20),x1,y1,x2,y2,t); tiles.append(tile(im,e,(x1,y1-20,x2,y2+20),f'{b[:6]} exit'))
rows=[np.hstack(tiles[i:i+4]+[np.full_like(tiles[0],255)]*(4-len(tiles[i:i+4]))) for i in range(0,len(tiles),4)]
cv2.imwrite(str(OUT/'thumbs'/'plant_check.jpg'),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,88])
