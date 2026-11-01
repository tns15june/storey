"""Sheet 057 (Block 1E, 1:50) and sheet 052 near the S1 leak, plus door pixels of the IFC mask near it. No API."""
import json, sys
sys.path.insert(0, "lab/agent")
from routelib import *
import draw, pymupdf as fitz
O = "data/lab/agent/mine/"
res = json.load(open(O + "routes.json"))
r = next(x for x in res["1"] if x["room"] == 100)
d = load(1)
pts = [tuple(q) for q in r["pts"]]
s = sample(pts)
free = cv2.dilate(d["gt"].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
off = np.nonzero(~free[s[:, 0], s[:, 1]])[0]
a, b = s[off[0]], s[off[-1]]
print("run ends", a, b)
door = d["gt"] & (d["lab"] == 0)
for q in (a, b):
    y, x = q; R = 60
    print(q, "door px within 1.5 m", int(door[y - R:y + R, x - R:x + R].sum()), "gt px", int(d["gt"][y - R:y + R, x - R:x + R].sum()))
pg = fitz.open(K.D + SHEET[1])[0]
y, x = b; px, py = x / K.SC + K.BOX[0], y / K.SC + K.BOX[1]
print("sheet pt", px, py)
R = 30
sc = 15
pm = pg.get_pixmap(matrix=fitz.Matrix(sc, sc), clip=fitz.Rect(px - R, py - R, px + R, py + R), alpha=False)
im = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, ::-1].copy()
P = lambda u, v: (int((u - px + R) * sc), int((v - py + R) * sc))
cv2.polylines(im, [np.array([P(q[1] / K.SC + K.BOX[0], q[0] / K.SC + K.BOX[1]) for q in pts], np.int32)], False, (255, 0, 255), 2)
cv2.imwrite(O + "zoom_s1_wall.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
# gt/solid view around the crossing
y0, y1, x0, x1 = y - 200, y + 200, x - 300, x + 300
v = np.full((400, 600, 3), 255, np.uint8)
v[d["gt"][y0:y1, x0:x1]] = (200, 240, 200); v[d["walk"][y0:y1, x0:x1] & ~d["gt"][y0:y1, x0:x1]] = (170, 200, 255); v[d["solid"][y0:y1, x0:x1]] = (40, 40, 40)
cv2.imwrite(O + "zoom_s1_mask.png", cv2.resize(v, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))
