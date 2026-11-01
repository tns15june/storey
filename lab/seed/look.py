import sys; sys.path.insert(0, "lab/seed")
from common import *
for bid in sys.argv[1:]:
    fl = CAT[bid]["floors"]
    ims = []
    for i, f in enumerate(fl):
        im = load_floor(bid, f)
        cv2.imwrite(str(OUT / "sheets" / f"{bid}__{i}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        s = cv2.resize(im, (980, round(im.shape[0] * 980 / im.shape[1])))
        cv2.putText(s, f"{i}: level {f['level']}", (10, 34), 0, 1.0, (0, 0, 255), 2)
        ims.append(s)
    h = max(x.shape[0] for x in ims)
    ims = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in ims]
    rows = [np.hstack(ims[i:i + 2] + ([np.full_like(ims[0], 255)] if len(ims[i:i + 2]) == 1 else [])) for i in range(0, len(ims), 2)]
    cv2.imwrite(str(OUT / "sheets" / f"{bid}_contact.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
