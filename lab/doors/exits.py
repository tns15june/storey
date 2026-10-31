"""Exterior doors. Geometry: flood the outside from the sheet edge with door openings closed, then test both sides of
each door. Spark: one whole-plan prompt per building. Truth: IFC IsExternal (clinic), door schedule (capitol), eye (admin, Brandeis)."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, fitz, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab"); sys.path.insert(0, str(Path(__file__).parent))
import key as K
from spark_tiles import SCHEMA, score, CACHE
import metalab as m

PROMPT = ('Point to every instance of the following object in this floor plan drawing and count them: "exterior door". '
          "An exterior door is a door in the outer wall of the building that leads directly from the inside to the outside. "
          "Do not include doors between two rooms, doors to corridors, or doors to stairs inside the building. A double door counts once. "
          'Point at the door opening. Answer in the format {"points": [{"x": <int>, "y": <int>}, ...], "count": <int>}. The coordinates should be in the 0-1000 range.')


def spark_exits(name, im, rect):
    f = CACHE / f"exit_{name}.json"
    if f.exists():
        o = json.loads(f.read_text())
    else:
        r, info = m.spark(PROMPT, images=[im], schema=SCHEMA)
        o = {"points": r["points"], "count": r["count"], "info": info}
        f.write_text(json.dumps(o))
    x0, y0, x1, y1 = rect
    return [[x0 + p["x"] / 1000 * (x1 - x0), y0 + p["y"] / 1000 * (y1 - y0)] for p in o["points"]], o


def page_img(path, p, dpi, clip=None):
    pix = fitz.open(str(path))[p].get_pixmap(dpi=dpi, clip=fitz.Rect(*clip) if clip else None)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3][:, :, ::-1].copy()


def outside_mask(im, z, doors, pt_per_m, seal_m=0.2, dark=170):
    """True where a point can reach the sheet border without crossing ink. z is px per pt, doors are (hinge, shut) in px."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ink = (g < dark).astype(np.uint8)
    k = max(3, int(2 * seal_m * pt_per_m * z) | 1)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    for h, s in doors:
        cv2.line(ink, tuple(int(v) for v in h), tuple(int(v) for v in s), 1, max(3, k // 2))
    flood = (1 - ink).astype(np.uint8)
    n, lab = cv2.connectedComponents(flood, connectivity=4)
    # the sheet frame seals the border, so the outside is also the largest free region
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    edge.add(1 + int(np.bincount(lab[lab > 0] - 1).argmax()))
    return np.isin(lab, list(edge))


def door_exterior(out, h, s, z, pt_per_m):
    """Both sides of the opening at 0.9 m. Exterior when exactly one side reaches the sheet edge."""
    h, s = np.array(h), np.array(s)
    mid, u = (h + s) / 2, (s - h) / np.linalg.norm(s - h)
    n = np.array([-u[1], u[0]]) * 0.9 * pt_per_m * z
    side = []
    for p in (mid + n, mid - n):
        x, y = int(p[0]), int(p[1])
        H, W = out.shape
        side.append(bool(out[np.clip(y - 3, 0, H - 1):y + 4, np.clip(x - 3, 0, W - 1):x + 4].any()))
    return side[0] != side[1]


def clinic(tag):
    A, L, arcs, reg = K.clinic_key(tag)
    leafy = [a for a in arcs if a["leaf"]]
    h = np.array([l["s_hinge"] for l in L])
    box = (h[:, 0].min() - 60, h[:, 1].min() - 60, h[:, 0].max() + 60, h[:, 1].max() + 60)
    zdpi = 110
    im = K.render(tag, zdpi)
    z = zdpi / 72
    doors = [((a["cx"] * z, a["cy"] * z), (a["shut_end"][0] * z, a["shut_end"][1] * z)) for a in leafy]
    out = outside_mask(im, z, doors, K.PT_PER_M)
    geo = [[(a["cx"] + a["shut_end"][0]) / 2, (a["cy"] + a["shut_end"][1]) / 2] for a in leafy
           if door_exterior(out, (a["cx"] * z, a["cy"] * z), (a["shut_end"][0] * z, a["shut_end"][1] * z), z, K.PT_PER_M)]
    # truth at door level: one point per IFC door on an external wall. Fence gates have z below the floor.
    truth = {}
    for l in L:
        if l["external"] and abs(l.get("z", 0)) >= 0:
            truth.setdefault(l["door"], []).append([(l["s_hinge"][0] + l["s_shut"][0]) / 2, (l["s_hinge"][1] + l["s_shut"][1]) / 2])
    T = [np.mean(v, axis=0).tolist() for v in truth.values()]
    gt = score(geo, T, 1.5 * K.PT_PER_M)
    vis = K.render(tag, 200 * 2 // 4 * 2, clip=fitz.Rect(*box))
    px = K.render(tag, 200, clip=fitz.Rect(*box))
    pts, o = spark_exits(f"clinic_{tag}", px, box)
    st = score(pts, T, 1.5 * K.PT_PER_M)
    zv = 200 / 72
    for p in T:
        cv2.circle(px, (int((p[0] - box[0]) * zv), int((p[1] - box[1]) * zv)), 22, (0, 170, 0), 3)
    for p in geo:
        cv2.drawMarker(px, (int((p[0] - box[0]) * zv), int((p[1] - box[1]) * zv)), (255, 0, 0), cv2.MARKER_SQUARE, 34, 2)
    for p in pts:
        cv2.drawMarker(px, (int((p[0] - box[0]) * zv), int((p[1] - box[1]) * zv)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 3)
    cv2.imwrite(str(K.OUT / f"exits_clinic_{tag}.png"), px)
    return {"truth_external_doors": len(T), "geometry": {"found": len(geo), "tp": gt}, "spark": {"found": len(pts), "count": o["count"], "tp": st, "usd": o["info"]["usd"]}}


def eye(name, path, p, box, truth, pt_per_m, dpi, tol_m=1.5):
    im = page_img(path, p, dpi, box)
    pts, o = spark_exits(name, im, box)
    st = score(pts, truth, tol_m * pt_per_m)
    z = dpi / 72
    vis = im.copy()
    for q in truth:
        cv2.circle(vis, (int((q[0] - box[0]) * z), int((q[1] - box[1]) * z)), 22, (0, 170, 0), 3)
    for q in pts:
        cv2.drawMarker(vis, (int((q[0] - box[0]) * z), int((q[1] - box[1]) * z)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 3)
    cv2.imwrite(str(K.OUT / f"exits_{name}.png"), vis)
    return {"truth_exterior_doors": len(truth), "spark": {"found": len(pts), "count": o["count"], "tp": st, "usd": o["info"]["usd"]}}


def main():
    res = {}
    for tag in K.SHEETS:
        res["clinic_" + tag] = clinic(tag)
        print(tag, res["clinic_" + tag], flush=True)
    lab = json.load(open(Path(__file__).parent / "labels_af_ces_admin.json"))
    T = [np.mean(v, axis=0).tolist() for v in lab["exterior"].values()]
    res["af_ces_admin"] = eye("af_ces_admin", K.ROOT / lab["sheet"], lab["page"], lab["plan_box"], T, lab["pt_per_m"], 250)
    print(res["af_ces_admin"], flush=True)
    res["mml_capitol"] = eye("mml_capitol", K.ROOT / "data/plans/mml_capitol/THE-CAPITOL.pdf", 3, CAP["box"], CAP["doors"], CAP["pt_per_m"], 150)
    print(res["mml_capitol"], flush=True)
    (K.OUT / "exits.json").write_text(json.dumps(res, indent=1))


CAP = None
if __name__ == "__main__":
    CAP = json.load(open(Path(__file__).parent / "labels_capitol.json"))
    main()
