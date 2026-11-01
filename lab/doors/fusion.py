"""Spark says where a door is, deterministic code measures it. Spark points gated onto raster arc candidates, using cached calls only."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "lab"); sys.path.insert(0, str(Path(__file__).parent))
import key as K
from spark_tiles import tiles, to_sheet, nms, score, arc_point, CACHE
from vector_eval import midpoint


def pick(sp, cand, tol):
    """Best raster candidate for each Spark point, each candidate used once."""
    c = np.array([[a["mx"], a["my"]] for a in cand])
    D = np.linalg.norm(np.array(sp)[:, None] - c[None], axis=2)
    r, i = linear_sum_assignment(np.minimum(D, tol + 1) - 0.01 * np.array([[a["score"] for a in cand]]) * tol)
    return [(a, b) for a, b in zip(r, i) if D[a, b] <= tol]


res = {}
for tag, name in (("F1", "t5low"),):
    A, L, arcs, reg = K.clinic_key(tag)
    G = [arc_point(l) for l in L]
    outs = [json.loads(f.read_text()) for f in sorted(CACHE.glob(f"{tag}_{name}_*.json"))]
    sp = nms([p for o in outs for p in to_sheet(o)], 0.45 * K.PT_PER_M)
    # vector and Spark union: recall a vector sheet could reach
    leafy = [a for a in arcs if a["leaf"]]
    vec = [[a["cx"] + (a["open_end"][0] + a["shut_end"][0] - 2 * a["cx"]) * .5, a["cy"] + (a["open_end"][1] + a["shut_end"][1] - 2 * a["cy"]) * .5] for a in leafy]
    union = nms(vec + sp, 0.5 * K.PT_PER_M)
    res["clinic_F1_vector_plus_spark"] = {"points": len(union), "tp": score(union, G, 0.8 * K.PT_PER_M), "truth": len(G)}
    cand = json.load(open(K.OUT / f"raster_cand_clinic_{tag}.json"))
    pairs = pick(sp, cand, 0.8 * K.PT_PER_M)
    got = [[cand[j]["mx"], cand[j]["my"]] for _, j in pairs]
    tp = score(got, G, 0.5 * K.PT_PER_M)
    # measurement quality on points that matched the key
    h = np.array([l["s_hinge"] for l in L])
    wid, side = [], 0
    for _, j in pairs:
        a = cand[j]
        d = np.linalg.norm(h - [a["cx"], a["cy"]], axis=1)
        i = d.argmin()
        if d[i] < 0.3 * K.PT_PER_M:
            l = L[i]
            wid.append(abs(a["r_pt"] / K.PT_PER_M - l["r"]) * 1000)
            sx, sy = ((1, 1), (-1, 1), (-1, -1), (1, -1))[int(a["q"])]
            u = (np.array(l["s_open"]) - l["s_hinge"]) + (np.array(l["s_shut"]) - l["s_hinge"])
            side += (np.sign(u[0]) == sx) and (np.sign(u[1]) == sy)
    res["clinic_F1_spark_gated_raster"] = {"spark_pts": len(sp), "raster_cand": len(cand), "kept": len(got), "tp": tp, "recall": tp / len(G), "precision": tp / max(len(got), 1),
                                           "hinge_within_0.3m": len(wid), "quadrant_agree": side / max(len(wid), 1), "width_err_mm_median": float(np.median(wid)), "width_err_mm_p95": float(np.percentile(wid, 95))}
    print(res)
LAB = json.load(open(Path(__file__).parent / "labels_af_ces_admin.json"))
G = LAB["interior"] + [p for v in LAB["exterior"].values() for p in v]
PM = LAB["pt_per_m"]
outs = [json.loads(f.read_text()) for f in sorted(CACHE.glob("admin_g6x3_*.json"))]
sp = nms([p for o in outs for p in to_sheet(o)], 0.45 * PM)
cand = json.load(open(K.OUT / "raster_cand_admin.json"))
pairs = pick(sp, cand, 0.8 * PM)
got = [[cand[j]["mx"], cand[j]["my"]] for _, j in pairs]
tp = score(got, G, 0.8 * PM)
rs = [cand[j]["r_pt"] / PM for _, j in pairs]
res["admin_spark_gated_raster"] = {"spark_pts": len(sp), "raster_cand": len(cand), "kept": len(got), "tp": tp, "truth": len(G), "recall": tp / len(G), "precision": tp / max(len(got), 1),
                                   "radius_m_median": float(np.median(rs)), "radius_m_p10": float(np.percentile(rs, 10)), "radius_m_p90": float(np.percentile(rs, 90))}
print(res["admin_spark_gated_raster"])
(K.OUT / "fusion.json").write_text(json.dumps(res, indent=1))
