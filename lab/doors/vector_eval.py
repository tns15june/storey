"""Vector arc detector against the IFC key: recall, precision, swing side, clear width. No API calls."""
import json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).parent))
import key as K

TOL_MID = 7.0  # pt, about 0.37 m, on the opening midpoint so a flipped hinge still counts as the same door


def match(P, Q, tol):
    if not len(P) or not len(Q):
        return []
    D = np.linalg.norm(np.asarray(P)[:, None] - np.asarray(Q)[None], axis=2)
    r, c = linear_sum_assignment(np.minimum(D, tol + 1))  # the cap makes it maximise matches, not minimise total distance
    return [(i, j) for i, j in zip(r, c) if D[i, j] <= tol]


def crops(tag, items, name):
    """Grid of crops around sheet points for eyeballing."""
    if not items:
        return
    z = 150 / 72
    im = K.render(tag, 150)
    tiles = []
    for (x, y, lab, col) in items[:80]:
        cx, cy = int(x * z), int(y * z)
        pad = 90
        c = im[max(cy - pad, 0):cy + pad, max(cx - pad, 0):cx + pad].copy()
        c = cv2.copyMakeBorder(c, 0, 180 - c.shape[0], 0, 180 - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.circle(c, (90, 90), 7, col, 1)
        cv2.putText(c, lab, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
        tiles.append(c)
    n = 10
    rows = [np.hstack(tiles[i:i + n] + [np.full((180, 180, 3), 255, np.uint8)] * (n - len(tiles[i:i + n]))) for i in range(0, len(tiles), n)]
    cv2.imwrite(str(K.OUT / name), np.vstack(rows))


def midpoint(h, sh):
    return [(h[0] + sh[0]) / 2, (h[1] + sh[1]) / 2]


def arc_mid(a):
    return midpoint([a["cx"], a["cy"]], a["shut_end"])


def match_arcs(pool, L, tol):
    """Same door if either hinge-to-arc-end chord has its midpoint on the true opening: the door exists whichever end the leaf logic chose."""
    if not pool:
        return []
    G = np.array([midpoint(l["s_hinge"], l["s_shut"]) for l in L])
    M = np.array([[midpoint([a["cx"], a["cy"]], a["a"]), midpoint([a["cx"], a["cy"]], a["b"])] for a in pool])
    D = np.linalg.norm(M[:, :, None] - G[None, None], axis=3).min(1)
    r, c = linear_sum_assignment(np.minimum(D, tol + 1))  # the cap makes it maximise matches, not minimise total distance
    return [(i, j) for i, j in zip(r, c) if D[i, j] <= tol]


def footprint(L):
    """Plan area: the box of the key's hinges plus a margin. shortcut: uses the key, a product needs the plan region."""
    h = np.array([l["s_hinge"] for l in L])
    return h.min(0) - 20, h.max(0) + 20


def in_box(a, box):
    p = np.array([a["cx"], a["cy"]])
    return bool(((p >= box[0]) & (p <= box[1])).all())


def quarter(a):
    return 80 <= a["sweep"] <= 100 and 0.65 * K.PT_PER_M <= a["r"] <= 1.3 * K.PT_PER_M


def main():
    res = {}
    for tag in K.SHEETS:
        A, L, arcs, reg = K.clinic_key(tag)
        G = [midpoint(l["s_hinge"], l["s_shut"]) for l in L]
        leafy = [a for a in arcs if a["leaf"]]
        row = {"reg": reg, "leaves": len(L), "doors": len({l["door"] for l in L})}
        box = footprint(L)
        qu = [a for a in arcs if in_box(a, box) and (a["leaf"] or quarter(a))]
        for name, pool in (("all_arcs", arcs), ("with_leaf", leafy), ("leaf_or_quarter_in_plan", qu)):
            m = match_arcs(pool, L, TOL_MID)
            row[name] = {"arcs": len(pool), "tp": len(m), "recall": len(m) / len(L), "precision": len(m) / len(pool)}
        leafy = qu
        m = match_arcs(leafy, L, TOL_MID)
        ti, tj = {i for i, _ in m}, {j for _, j in m}
        fp = [i for i in range(len(leafy)) if i not in ti]
        fn = [j for j in range(len(L)) if j not in tj]
        same_hinge = swing_ok = 0
        werr = []
        for i, j in m:
            a, l = leafy[i], L[j]
            h = np.array([a["cx"], a["cy"]])
            same_hinge += np.linalg.norm(h - l["s_hinge"]) < 2 * TOL_MID
            u, w = np.array(a["open_end"]) - h, np.array(l["s_open"]) - l["s_hinge"]
            swing_ok += u @ w > 0
            werr.append(abs(a["r"] / K.PT_PER_M - l["r"]) * 1000)
        row.update({"hinge_agree": same_hinge / len(m), "swing_side_agree": swing_ok / len(m), "n_tp": len(m),
                    "width_err_mm_mean": float(np.mean(werr)), "width_err_mm_p95": float(np.percentile(werr, 95)), "width_err_mm_max": float(np.max(werr)),
                    "fn_types": {n: sum(1 for j in fn if L[j]["name"] == n) for n in {L[j]["name"] for j in fn}}})
        res[tag] = row
        crops(tag, [(*arc_mid(leafy[i]), f"FP{i} r{leafy[i]['r']/K.PT_PER_M:.2f} s{leafy[i]['sweep']:.0f}", (0, 0, 255)) for i in fp], f"fp_{tag}.png")
        crops(tag, [(*G[j], f"FN {L[j]['name'][2:12]}", (255, 0, 0)) for j in fn], f"fn_{tag}.png")
        print(tag, json.dumps(row))
    (K.OUT / "vector_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
