"""Zoom a route candidate at the points where it enters and leaves the IFC walkable mask. No API."""
import json, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/agent")
from routelib import *
import draw, pymupdf as fitz
O = "/Users/tsharma/Github/Meta/data/lab/agent/mine/"
res = json.load(open(O + "routes.json"))
fl, room = int(sys.argv[1]), int(sys.argv[2])
r = next(x for x in res[str(fl)] if x["room"] == room)
d = load(fl); pg = fitz.open(K.D + SHEET[fl])[0]
pts = [tuple(q) for q in r["pts"]]
s = sample(pts)
free = cv2.dilate(d["gt"].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
off = ~free[s[:, 0], s[:, 1]]
lab, n = ndi.label(off)
ims = []
for i in range(1, n + 1):
    k = np.nonzero(lab == i)[0]
    if len(k) < 8:
        continue
    for m in (k[0], k[-1]):
        y, x = s[m]; px, py = x / K.SC + K.BOX[0], y / K.SC + K.BOX[1]
        R = 45
        im, P, sc = draw.render(pg, (px - R, py - R, px + R, py + R), long=620, max_dpi=500)
        cv2.polylines(im, [np.array([P(q[1] / K.SC + K.BOX[0], q[0] / K.SC + K.BOX[1]) for q in pts], np.int32)], False, (255, 0, 255), 2)
        # IFC walkable tint
        h, w = im.shape[:2]
        ims.append(cv2.copyMakeBorder(im, 0, 620 - h, 0, 620 - w, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
cv2.imwrite(O + f"zoom_{fl}_{room}.jpg", np.hstack(ims[:3]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(len(ims), r["cross"])
