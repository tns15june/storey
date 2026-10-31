import sys, json, cv2, numpy as np, fitz
from pathlib import Path
R=Path('data/plans'); O=Path('data/lab/defects/thumbs')
c={e['id']:e for e in json.load(open(R/'catalog.json'))}
for b in ['schenley','bme_f','bme_k','habs_va_bldg1','habs_stelizabeths_atkins','habs_stelizabeths_center','mml_capitol','mml_baxter','nibs_clinic']:
    e=c[b]; ims=[]
    for f in e['floors']:
        p=R/b/f['file']
        if p.suffix.lower()=='.pdf':
            d=fitz.open(p); pg=d[f['page']]; z=2000/max(pg.rect.width,pg.rect.height); px=pg.get_pixmap(matrix=fitz.Matrix(z,z)); im=np.frombuffer(px.samples,np.uint8).reshape(px.h,px.w,px.n)[...,:3][...,::-1].copy()
        else: im=cv2.imread(str(p))
        print(b,f['level'],im.shape)
        s=700/max(im.shape[:2]); ims.append(cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_AREA))
    h=max(i.shape[0] for i in ims); ims=[cv2.copyMakeBorder(i,0,h-i.shape[0],0,4,cv2.BORDER_CONSTANT,value=(0,0,255)) for i in ims]
    cv2.imwrite(str(O/f'{b}.jpg'),np.hstack(ims))
