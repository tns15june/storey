"""Step 2. lab/align/reg.py blocks on every floor against the reference floor, with its inlier gate, plus overlays to read. No API call.
A floor with no inlier count or under the gate is written as unaligned. Truth boxes are used only to report the error, never to decide the gate."""
import json
import sys

import cv2
import numpy as np

import sys; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/presence")
import pres
from pres import OUT, T, GATE, reg, ctr

pres.load_spark()
(OUT / "align").mkdir(exist_ok=True)
(OUT / "overlays").mkdir(exist_ok=True)
for b in (sys.argv[1:] or list(T)):
    ref = pres.ref_of(b)
    R = reg.sheet(b, ref)
    x1, y1, x2, y2 = reg.plan_box(b, ref, R.shape)
    rows = {}
    for f in range(pres.nfloors(b)):
        if f == ref:
            rows[f] = dict(H=np.eye(3).tolist(), inliers=None, matches=None, aligned=True, is_ref=True)
            continue
        c = reg.context(b, f, ref)
        H = reg.blocks(c)
        inl = c.get("inliers")
        ok = inl is not None and inl >= GATE
        s = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
        rows[f] = dict(H=H.tolist(), inliers=inl, matches=c.get("matches"), aligned=ok, scale=round(s, 4), rot=round(float(np.degrees(np.arctan2(H[1, 0], H[0, 0]))), 2), shift=[round(float(H[0, 2]), 1), round(float(H[1, 2]), 1)])
        a = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
        g = reg.warp(cv2.cvtColor(reg.sheet(b, f), cv2.COLOR_BGR2GRAY), H, a.shape)
        cv2.imwrite(str(OUT / f"overlays/{b}_f{f}.jpg"), cv2.merge([a, a, g])[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 85])
    # error of objects drawn on both this floor and the reference floor, for the report only
    for f, r in rows.items():
        if f == ref:
            continue
        H = np.array(r["H"])
        e = [float(np.linalg.norm(pres.apply_pt(H, ctr(v["boxes"][str(f)])) - np.array(ctr(v["boxes"][str(ref)]))))
             for v in T[b]["objects"].values() if str(f) in v["boxes"] and str(ref) in v["boxes"]]
        r["truth_err_px"] = dict(n=len(e), median=round(float(np.median(e)), 1), max=round(float(max(e)), 1)) if e else None
        print(f"{b:26s} f{f} inliers {r['inliers']}/{r['matches']} aligned {r['aligned']} scale {r['scale']} rot {r['rot']} shift {r['shift']} err {r['truth_err_px']}", flush=True)
    json.dump({str(k): v for k, v in rows.items()}, open(OUT / f"align/{b}.json", "w"), indent=1)
