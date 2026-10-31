"""Overlays of the real route crossings on the drawing. No API."""
import json, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/agent")
from routelib import *
import draw, pymupdf as fitz
O = "/Users/tsharma/Github/Meta/data/lab/agent/mine/"
res = json.load(open(O + "routes.json"))
fl = 1
pg = fitz.open(K.D + SHEET[fl])[0]
print(pg.rect, pg.rotation)
d = load(fl)
tiles = []
for r in [x for x in res["1"] if x["cross"]]:
    pts = [(q[1] / K.SC + K.BOX[0], q[0] / K.SC + K.BOX[1]) for q in r["pts"]]
    a = np.array(pts); m = 3 * K.PPM
    clip = (a[:, 0].min() - m, a[:, 1].min() - m, a[:, 0].max() + m, a[:, 1].max() + m)
    im, P, s = draw.render(pg, clip)
    draw.route(im, P, pts)
    for y, x, n in r["cross"]:
        q = P(x / K.SC + K.BOX[0], y / K.SC + K.BOX[1]); cv2.circle(im, q, 28, (0, 200, 255), 2)
    cv2.imwrite(O + f"real_route_{r['room']}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(r["room"], r["m"], r["cross"], im.shape)
