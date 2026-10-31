"""Every place a good route goes between IFC spaces, zoomed on the sheet, to check the labels the models disputed. No API."""
import json, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/agent")
from routelib import *
import pymupdf as fitz
O = "/Users/tsharma/Github/Meta/data/lab/agent/mine/"
res = json.load(open(O + "routes.json"))
ims = []
for fl, room in ((1, 8), (1, 29), (2, 8), (2, 55)):
    r = next(x for x in res[str(fl)] if x["room"] == room)
    d = load(fl); pg = fitz.open(K.D + SHEET[fl])[0]
    pts = [tuple(q) for q in r["pts"]]
    s = sample(pts)
    between = d["lab"][s[:, 0], s[:, 1]] == 0
    lab_, n = ndi.label(between)
    spots = []
    for i in range(1, n + 1):
        k = np.nonzero(lab_ == i)[0]
        spots.append(s[k[len(k) // 2]])
    # both route ends too
    spots = spots + [s[0], s[-1]]
    print(fl, room, r["m"], "passages", n, [tuple(int(v) for v in q) for q in spots])
    for q in spots:
        y, x = q; px, py = x / K.SC + K.BOX[0], y / K.SC + K.BOX[1]
        R = 34; sc = 620 / (2 * R)
        pm = pg.get_pixmap(matrix=fitz.Matrix(sc, sc), clip=fitz.Rect(px - R, py - R, px + R, py + R), alpha=False)
        im = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, ::-1].copy()
        P = lambda u, v: (int((u - px + R) * sc), int((v - py + R) * sc))
        cv2.polylines(im, [np.array([P(a[1] / K.SC + K.BOX[0], a[0] / K.SC + K.BOX[1]) for a in pts], np.int32)], False, (255, 0, 255), 2)
        cv2.putText(im, f"F{fl} r{room}", (4, 16), 0, .5, (0, 0, 255), 1)
        ims.append(cv2.copyMakeBorder(im, 0, 622 - im.shape[0], 0, 622 - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)))
c = 4
while len(ims) % c: ims.append(np.full_like(ims[0], 255))
rows = [np.hstack(ims[i:i + c]) for i in range(0, len(ims), c)]
for i in range(0, len(rows), 3):
    cv2.imwrite(O + f"zoom3_{i // 3}.jpg", np.vstack(rows[i:i + 3]), [cv2.IMWRITE_JPEG_QUALITY, 86])
print(len(ims))
# the number in value_agree_0 and the small stair in stair_good_bme__14_0
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/modern")
pg = fitz.open(K.D + "2012-03-27-ClinicRedactedDrawings/Clinic_055_First Floor Plan Block 1C.pdf")[0]
x, y = 1780.5, 747.4; R = 32; sc = 16
pm = pg.get_pixmap(matrix=fitz.Matrix(sc, sc), clip=fitz.Rect(x - 18, y - R, x + 18, y + R), alpha=False)
im = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, ::-1].copy()
cv2.imwrite(O + "zoom_2430.png", cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE))
im = cv2.imread("/Users/tsharma/Github/Meta/data/lab/audit/bme_k/f0.png")
c = im[770:870, 780:880]
cv2.imwrite(O + "zoom_curve.png", cv2.resize(c, None, fx=7, fy=7, interpolation=cv2.INTER_CUBIC))
