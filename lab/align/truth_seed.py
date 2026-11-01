"""Seed floor-to-floor landmark correspondences by local patch search from the coarse init, and draw galleries to check by eye.
Rows are landmarks, columns floors. The crosshair is the seed. Verified points go in truth_fixes.py; this file only proposes."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "lab/align")
import reg

REF = {"bme_f": 1, "bme_k": 1, "schenley": 1, "habs_stelizabeths_center": 1, "habs_stelizabeths_atkins": 1, "habs_va_bldg1": 1,
       "mml_capitol": 1, "mml_baxter": 0, "wfu_efird": 1, "nibs_clinic": 0}
STAIRS = {  # rough hand picks on the ref floor from the contact sheets, 2000 px long side
    "bme_f": [(472, 250), (1316, 256), (566, 694), (1228, 694), (894, 938)],
    "bme_k": [(825, 734), (1144, 734), (300, 719), (1669, 728), (994, 275)],
    "schenley": [(1009, 125), (284, 922), (1753, 922)],
    "habs_stelizabeths_center": [(353, 906), (1669, 906), (903, 766)],
    "habs_stelizabeths_atkins": [(700, 594)],
    "habs_va_bldg1": [],
    "mml_capitol": [(1425, 547)],
    "mml_baxter": [],
    "wfu_efird": [(811, 860)],
    "nibs_clinic": [(890, 850), (1078, 390)],
}


def autos(c, nx=4, ny=3):
    x1, y1, x2, y2 = c["ref_box"]
    thick = cv2.morphologyEx((c["ref_ink"] > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.float32)
    g = cv2.GaussianBlur(thick, (0, 0), 2.5)
    out = []
    for j in range(ny):
        for i in range(nx):
            xa, xb = x1 + (x2 - x1) * i // nx, x1 + (x2 - x1) * (i + 1) // nx
            ya, yb = y1 + (y2 - y1) * j // ny, y1 + (y2 - y1) * (j + 1) // ny
            p = cv2.goodFeaturesToTrack(g[ya:yb, xa:xb], 1, 0.01, 20, blockSize=15)
            if p is not None:
                out.append((int(p[0, 0, 0]) + xa, int(p[0, 0, 1]) + ya))
    return out


def seed(bid):
    ref = REF[bid]
    n = len(reg.INDEX[bid])
    base = reg.context(bid, ref, ref)
    lms = [{"name": f"stair{k}", "p": list(p), "kind": "stair"} for k, p in enumerate(STAIRS[bid])] + [{"name": f"corner{k}", "p": list(p), "kind": "corner"} for k, p in enumerate(autos(base))]
    res = {l["name"]: {"kind": l["kind"], "ref": l["p"], "floors": {}} for l in lms}
    for f in range(n):
        if f == ref:
            continue
        c = reg.context(bid, f, ref)
        H = reg.coarse(c)
        mw = reg.warp(c["mov_soft"], H, c["ref_soft"].shape, 0)
        Hinv = np.linalg.inv(H)
        for l in lms:
            x, y = l["p"]
            p, s = 70, 110
            if not (p <= x < mw.shape[1] - p and p <= y < mw.shape[0] - p):
                continue
            t = c["ref_soft"][y - p:y + p, x - p:x + p]
            ya, yb, xa, xb = max(0, y - p - s), min(mw.shape[0], y + p + s), max(0, x - p - s), min(mw.shape[1], x + p + s)
            if t.std() < 1e-3:
                continue
            r = cv2.matchTemplate(mw[ya:yb, xa:xb], t, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(r)
            q = reg.apply(Hinv, [(xa + loc[0] + p, ya + loc[1] + p)])[0]
            res[l["name"]]["floors"][str(f)] = {"xy": [round(float(q[0]), 1), round(float(q[1]), 1)], "ncc": round(float(v), 3)}
    return res


def gallery(bid, res, path, half=100, zoom=1.6):
    n = len(reg.INDEX[bid])
    ims = [reg.sheet(bid, i) for i in range(n)]
    rows = []
    S = int(2 * half * zoom)
    for name, l in res.items():
        tiles = []
        for f in range(n):
            if f == REF[bid]:
                xy = l["ref"]; lab = "REF"
            elif str(f) in l["floors"]:
                xy = l["floors"][str(f)]["xy"]; lab = str(l["floors"][str(f)]["ncc"])
            else:
                xy = None
            t = np.full((S, S, 3), 200, np.uint8)
            if xy is not None:
                x, y = int(round(xy[0])), int(round(xy[1]))
                pad = cv2.copyMakeBorder(ims[f], half, half, half, half, cv2.BORDER_CONSTANT, value=(180, 180, 180))
                t = cv2.resize(pad[y:y + 2 * half, x:x + 2 * half], (S, S), interpolation=cv2.INTER_CUBIC)
                cv2.line(t, (S // 2 - 14, S // 2), (S // 2 + 14, S // 2), (0, 0, 255), 1)
                cv2.line(t, (S // 2, S // 2 - 14), (S // 2, S // 2 + 14), (0, 0, 255), 1)
                cv2.putText(t, f"{name} f{f} {lab}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 0, 0), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    out = reg.OUT / "truth"
    out.mkdir(exist_ok=True)
    for bid in REF:
        r = seed(bid)
        json.dump(r, open(out / f"{bid}.seed.json", "w"), indent=1)
        gallery(bid, r, out / f"{bid}.jpg")
        print(bid, len(r), np.round([v["ncc"] for l in r.values() for v in l["floors"].values()], 2).tolist()[:12], flush=True)
