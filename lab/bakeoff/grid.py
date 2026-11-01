"""Throwaway viewer: crop in 0-1000 sheet coordinates with a labelled grid, to set truth by eye."""
import sys, cv2
sys.path.insert(0, 'lab/bakeoff')
import sheets as s
def crop(key, x0, y0, x1, y1, name, step=50, px=6000, width=1500):
    im = s.full(key, px); h, w = im.shape[:2]
    c = im[y0*h//1000:y1*h//1000, x0*w//1000:x1*w//1000].copy()
    z = width / c.shape[1]; c = cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_AREA)
    for g in range((x0//step+1)*step, x1, step):
        X = int((g-x0)*w/1000*z); cv2.line(c, (X,0), (X,c.shape[0]), (0,160,255), 1); cv2.putText(c, str(g), (X+2,14), 0, .5, (0,0,255), 1)
    for g in range((y0//step+1)*step, y1, step):
        Y = int((g-y0)*h/1000*z); cv2.line(c, (0,Y), (c.shape[1],Y), (0,160,255), 1); cv2.putText(c, str(g), (2,Y-2), 0, .5, (0,0,255), 1)
    cv2.imwrite(str(s.OUT/'view'/f'{name}.png'), c)
if __name__ == '__main__':
    a = sys.argv; crop(a[1], *map(int, a[2:6]), a[6], int(a[7]) if len(a) > 7 else 50)
