"""Throwaway viewer: bottom-right title crops for setting truth by eye."""
import sys, cv2, numpy as np
sys.path.insert(0, 'lab/bakeoff')
import sheets as s
keys = [k for k in s.ALL if s.ALL[k][1] is not None]
V = s.OUT / 'view'
def cell(k):
    im = s.full(k, 7000); h, w = im.shape[:2]
    c = im[int(h*.70):, int(w*.72):]          # bottom right
    c = cv2.resize(c, (760, int(760*c.shape[0]/c.shape[1])), interpolation=cv2.INTER_AREA)
    c = cv2.copyMakeBorder(c, 26, 4, 4, 4, cv2.BORDER_CONSTANT, value=(220,220,255))
    cv2.putText(c, k, (6, 20), 0, .6, (0,0,200), 2); return c
for b in range(0, len(keys), 6):
    cs = [cell(k) for k in keys[b:b+6]]
    H = max(x.shape[0] for x in cs)
    cs = [cv2.copyMakeBorder(x, 0, H-x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255)) for x in cs]
    while len(cs) < 6: cs.append(np.full_like(cs[0], 255))
    cv2.imwrite(str(V / f'tb{b//6}.png'), np.vstack([np.hstack(cs[i:i+3]) for i in (0,3)]))
    print(keys[b:b+6])
