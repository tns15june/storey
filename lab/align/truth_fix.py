"""Turn the seeds into verified landmark truth. Every drop and extra below comes from reading the gallery crops by eye."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg
import truth_seed as ts

ALL = "*"
DROP = {
    "bme_f": {"stair1": [0]},                        # patch matched a different stair on f0, curved wall sits 15 px off
    "bme_k": {},
    "schenley": {"corner6": ALL, "corner4": ALL, "stair0": ALL},  # interior wings differ between floors
    "habs_va_bldg1": {"corner1": ALL, "corner6": ALL, "corner18": [4], "corner0": [4]},  # section letter, along-wall ambiguity, roof sheet differs
    "mml_capitol": {"stair0": ALL, "corner7": ALL, "corner4": [2]},  # stair0 wrong room; corner7 and corner4 f2 slid along a wall: orb_mask, ecc_affine and blocks all agree 50 px off the seed and the overlay shows the walls coincide
    "nibs_clinic": {"corner6": ALL, "stair0": ALL},  # no matching structure on f1
    "wfu_efird": {"corner18": [3], "corner19": [3], "corner11": [3], "corner2": [3], "corner16": [3], "corner6": [3], "corner9": [3]},  # one room off along the corridor
    "habs_stelizabeths_center": {"corner19": ALL, "corner15": ALL, "corner6": ALL, "corner8": ALL, "corner16": ALL, "corner18": ALL,   # key plan and dimension text, fixed to the sheet
                                 "corner10": [0], "corner4": [0], "stair0": [4], "stair1": [4]},
}
EXTRA = {  # ref point, then per-floor rough guess offset from the ref point, refined by local NCC
    "wfu_efird": {"anchor": "stair0", "search": 45, "pts": {"room124_tl": (504, 605), "room104_bl": (504, 1115), "room121_tr": (897, 598), "lounge_br": (897, 1120)}},
    "habs_stelizabeths_center": {"anchor": None, "search": 45, "pts": {"pav_tl": (831, 212), "pav_tr": (1190, 212), "wing_nw": (144, 244), "wing_ne": (1878, 300), "mid_w": (660, 700), "mid_e": (1190, 700)}},
}


def near(c, p, guess, half=60, search=45, soft="soft"):
    x, y = p
    t = c["ref_" + soft][y - half:y + half, x - half:x + half]
    gx, gy = int(guess[0]), int(guess[1])
    m = c["mov_" + soft]
    ya, yb, xa, xb = max(0, gy - half - search), min(m.shape[0], gy + half + search), max(0, gx - half - search), min(m.shape[1], gx + half + search)
    r = cv2.matchTemplate(m[ya:yb, xa:xb], t, cv2.TM_CCOEFF_NORMED)
    _, v, _, loc = cv2.minMaxLoc(r)
    return [xa + loc[0] + half, ya + loc[1] + half], round(float(v), 3)


def grey_soft(im):
    # HABS walls are grey fill, so match on the grey level itself, not the ink threshold
    return cv2.GaussianBlur(255 - cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (0, 0), 1.2).astype(np.float32) / 255


out = {}
for bid, drops in DROP.items():
    seed = json.load(open(reg.OUT / f"truth/{bid}.seed.json"))
    ref = ts.REF[bid]
    res = {}
    for name, l in seed.items():
        d = drops.get(name)
        if d == ALL:
            continue
        floors = {f: v for f, v in l["floors"].items() if d is None or int(f) not in d}
        if floors:
            res[name] = dict(l, floors=floors)
    ex = EXTRA.get(bid)
    if ex:
        for name, p in ex["pts"].items():
            res[name] = {"kind": "corner", "ref": list(p), "floors": {}}
        for f in range(len(reg.INDEX[bid])):
            if f == ref:
                continue
            c = reg.context(bid, f, ref)
            c["ref_grey"], c["mov_grey"] = grey_soft(reg.sheet(bid, ref)), grey_soft(reg.sheet(bid, f))
            off = np.zeros(2)
            if ex["anchor"]:
                off = np.array(seed[ex["anchor"]]["floors"][str(f)]["xy"]) - np.array(seed[ex["anchor"]]["ref"])
            for name, p in ex["pts"].items():
                xy, v = near(c, p, np.array(p) + off, search=ex["search"], soft="grey")
                res[name]["floors"][str(f)] = {"xy": xy, "ncc": v}
    if bid == "habs_stelizabeths_center":  # wall differs on the two top floors at this point
        for f in ("3", "4"):
            res["mid_e"]["floors"].pop(f)
    out[bid] = res
    print(bid, len(res), "floor-pairs", sum(len(v["floors"]) for v in res.values()), flush=True)
    if ex:
        ts.gallery(bid, {k: res[k] for k in ex["pts"]}, reg.OUT / f"truth/{bid}.extra.jpg", half=75, zoom=1.4)
json.dump(out, open(reg.OUT / "truth/truth.json", "w"), indent=1)
