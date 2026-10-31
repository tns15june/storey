"""Align each building's floors to a reference sheet at 2000 px. Saves frames and align.json. No API calls."""
import sys; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed")
from common import *

REF = {"bme_f": 1, "schenley": 2, "habs_stelizabeths_atkins": 1, "habs_stelizabeths_center": 1}
for bid in sys.argv[1:] or ["bme_f", "schenley", "habs_stelizabeths_atkins", "habs_stelizabeths_center"]:
    d = OUT / bid; d.mkdir(exist_ok=True)
    fl = CAT[bid]["floors"]
    raw = [load_floor(bid, f) for f in fl]
    ref = raw[REF[bid]]
    info = {}
    for i, im in enumerate(raw):
        al, m = (ref, {"method": "ref", "overlap": 1.0}) if i == REF[bid] else align(im, ref)
        cv2.imwrite(str(d / f"f{i}.png"), al)
        info[i] = m
        print(bid, i, fl[i]["level"], m["method"], m["overlap"], {k: round(overlap(cv2.warpAffine(im, np.float32([[1,0,0],[0,1,0]]), (ref.shape[1], ref.shape[0]), borderValue=(235,235,235)), ref), 3) for k in ["identity"]})
    json.dump(info, open(d / "align.json", "w"), indent=1)
    # red = this floor, cyan = reference, to eyeball registration
    tiles = []
    for i in range(len(fl)):
        a = cv2.cvtColor(cv2.imread(str(d / f"f{i}.png")), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        tiles.append(cv2.resize(cv2.merge([b, b, a]), (980, round(a.shape[0] * 980 / a.shape[1]))))
    rows = [np.hstack(tiles[j:j + 2] + ([np.full_like(tiles[0], 255)] if len(tiles[j:j + 2]) == 1 else [])) for j in range(0, len(tiles), 2)]
    cv2.imwrite(str(d / "registration.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
