import json, sys
import cv2, numpy as np
sys.path.insert(0,'/Users/tsharma/Github/Meta/lab/defects')
from seeds import SEEDS
R='/Users/tsharma/Github/Meta/data/lab/'
orb=cv2.ORB_create(20000)
def fit(src,dst):
    ka,da=orb.detectAndCompute(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY),None); kb,db=orb.detectAndCompute(cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY),None)
    m=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(da,db)
    a=np.float32([ka[x.queryIdx].pt for x in m]); b=np.float32([kb[x.trainIdx].pt for x in m])
    A,inl=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=20000)
    return np.vstack([A,[0,0,1]]),int(inl.sum())
b='habs_va_bldg1'
out={}
tiles=[]
for i in range(5):
    st=cv2.imread(R+f'defects/stack/{b}/{i}.png'); raw=cv2.imread(R+f'presence/sheets/{b}/{i}.jpg')
    T,n=fit(st,raw)
    row=[]
    for o,bx in SEEDS[b]['objects'].items():
        p=np.array([[bx[1][0],bx[1][1],1],[bx[1][2],bx[1][3],1]],float)@T.T
        out.setdefault(o,{})[i]=[int(round(v)) for v in p[:,:2].ravel()]
    print(i,n,{k:v[i] for k,v in out.items()})
    e=out['elev1'][i]; f=out['elev2'][i]
    x1,y1,x2,y2=min(e[0],f[0])-60,min(e[1],f[1])-60,max(e[2],f[2])+60,max(e[3],f[3])+60
    c=raw[y1:y2,x1:x2].copy()
    for q in (e,f): cv2.rectangle(c,(q[0]-x1,q[1]-y1),(q[2]-x1,q[3]-y1),(0,160,0),1)
    tiles.append(cv2.resize(c,(320,int(320*c.shape[0]/c.shape[1])),interpolation=cv2.INTER_CUBIC))
h=max(t.shape[0] for t in tiles)
tiles=[cv2.copyMakeBorder(t,0,h-t.shape[0],0,4,cv2.BORDER_CONSTANT,value=(0,0,0)) for t in tiles]
cv2.imwrite(R+'presence/truth_views/z_va_elev.jpg',np.hstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,88])
json.dump(out,open(R+'presence/_va_elev.json','w'))
