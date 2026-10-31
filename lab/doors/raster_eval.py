"""Classical raster arc detector on the clinic (rendered) and the admin sheet (raster tiles), against the keys. No API calls."""
import json, sys
from pathlib import Path
import cv2, fitz, numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).parent))
import key as K, raster_arcs as R
from spark_tiles import arc_point, score

res = {}
for dpi in (150,):
    for tag in K.SHEETS:
        A, L, arcs, reg = K.clinic_key(tag)
        G = [arc_point(l) for l in L]
        h = np.array(G)
        box = (h[:, 0].min() - 30, h[:, 1].min() - 30, h[:, 0].max() + 30, h[:, 1].max() + 30)
        im = K.render(tag, dpi, clip=fitz.Rect(*box))
        z = dpi / 72
        f = R.find(im, 0.72 * K.PT_PER_M * z, 1.3 * K.PT_PER_M * z, thr=0.6, step=max(1, 0.05 * K.PT_PER_M * z))
        pts = [[box[0] + R.arc_mid(a)[0] / z, box[1] + R.arc_mid(a)[1] / z] for a in f]
        tp = score(pts, G, 0.4 * K.PT_PER_M)
        (K.OUT / f"raster_cand_clinic_{tag}.json").write_text(json.dumps([{**{k: float(v) for k, v in a.items()}, "cx": box[0] + a["x"] / z, "cy": box[1] + a["y"] / z, "r_pt": a["r"] / z, "mx": p[0], "my": p[1]} for a, p in zip(f, pts)]))
        res[f"clinic_{tag}_{dpi}dpi"] = {"detections": len(pts), "truth": len(G), "tp": tp, "recall": tp / len(G), "precision": tp / max(len(pts), 1)}
        print(f"clinic_{tag}_{dpi}dpi", res[f"clinic_{tag}_{dpi}dpi"], flush=True)
        if dpi == 150:
            vis = im.copy()
            for p in pts:
                cv2.circle(vis, (int((p[0] - box[0]) * z), int((p[1] - box[1]) * z)), 12, (0, 0, 255), 2)
            for g in G:
                cv2.circle(vis, (int((g[0] - box[0]) * z), int((g[1] - box[1]) * z)), 6, (0, 170, 0), 2)
            cv2.imwrite(str(K.OUT / f"raster_{tag}.png"), vis)
LAB = json.load(open(Path(__file__).parent / "labels_af_ces_admin.json"))
b = LAB["plan_box"]
pg = fitz.open(str(K.ROOT / LAB["sheet"]))[LAB["page"]]
G = LAB["interior"] + [p for v in LAB["exterior"].values() for p in v]
for dpi in (300,):
    px = pg.get_pixmap(dpi=dpi, clip=fitz.Rect(*b))
    im = np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, px.n)[:, :, :3][:, :, ::-1].copy()
    z = dpi / 72
    f = R.find(im, 0.75 * LAB["pt_per_m"] * z, 1.1 * LAB["pt_per_m"] * z, thr=0.6, step=1)
    pts = [[b[0] + R.arc_mid(a)[0] / z, b[1] + R.arc_mid(a)[1] / z] for a in f]
    tp = score(pts, G, 8)
    (K.OUT / "raster_cand_admin.json").write_text(json.dumps([{**{k: float(v) for k, v in a.items()}, "cx": b[0] + a["x"] / z, "cy": b[1] + a["y"] / z, "r_pt": a["r"] / z, "mx": p[0], "my": p[1]} for a, p in zip(f, pts)]))
    res[f"admin_{dpi}dpi"] = {"detections": len(pts), "truth": len(G), "tp": tp, "recall": tp / len(G), "precision": tp / max(len(pts), 1)}
    print(f"admin_{dpi}dpi", res[f"admin_{dpi}dpi"], flush=True)
    vis = im.copy()
    for p in pts:
        cv2.circle(vis, (int((p[0] - b[0]) * z), int((p[1] - b[1]) * z)), 14, (0, 0, 255), 2)
    for g in G:
        cv2.circle(vis, (int((g[0] - b[0]) * z), int((g[1] - b[1]) * z)), 8, (0, 170, 0), 2)
    cv2.imwrite(str(K.OUT / f"raster_admin_{dpi}.png"), vis)
(K.OUT / "raster_eval.json").write_text(json.dumps(res, indent=1))
