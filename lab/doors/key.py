"""Clinic answer key on the sheet: IFC swing leaves in sheet points, registered with the vector arcs."""
import json, sys
from pathlib import Path
import cv2, fitz, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import align, vec_arcs as v

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/lab/doors"
SHEETS = {"F1": ("First Floor", "Clinic_052_Composite First Floor Plan", (-1, 2.5)), "F2": ("Second Floor", "Clinic_059_Composite Second Floor Plan", (2.5, 7))}
DIR = ROOT / "data/plans/nibs_clinic/2012-03-27-ClinicRedactedDrawings"
PT_PER_M = align.SCALE


def page(tag):
    return fitz.open(str(DIR / (SHEETS[tag][1] + ".pdf")))[0]


def dedupe(arcs):
    """Same arc drawn twice (line weights, layers): keep one per centre and radius."""
    kept = []
    for A in sorted(arcs, key=lambda x: (x["leaf"] is None, -x["sweep"])):
        if not any(abs(A["cx"] - B["cx"]) < 1 and abs(A["cy"] - B["cy"]) < 1 and abs(A["r"] - B["r"]) < 0.5 for B in kept):
            kept.append(A)
    return kept


def vector_arcs(tag, lo=0.5, hi=1.7):
    s = v.page_segments(page(tag))
    a = v.attach_leaf(v.find_arcs(s, lo * PT_PER_M, hi * PT_PER_M), s)
    return dedupe(a)


def clinic_key(tag):
    """(A, leaves) with hinge, open and shut in sheet points. Doors with no storey are assigned by z."""
    doors = json.load(open(OUT / "clinic_ifc_doors.json"))
    st, _, (z0, z1) = SHEETS[tag]
    L, unassigned = [], 0
    for d in doors:
        keep = d["storey"] == st or (d["storey"] not in ("First Floor", "Second Floor") and z0 <= d["z"] < z1)
        if keep:
            for l in d["leaves"]:
                L.append({**l, "door": d["id"], "name": d["name"].split(":")[0], "external": d["external"], "width": d["width"], "storey": d["storey"]})
    # the IFC repeats some curtain wall doors at one spot
    seen, uniq = set(), []
    for l in L:
        k = (round(l["hinge"][0], 2), round(l["hinge"][1], 2))
        if k not in seen:
            seen.add(k); uniq.append(l)
    L = uniq
    arcs = vector_arcs(tag)
    I = np.array([l["hinge"] for l in L])
    H = np.array([[x["cx"], x["cy"]] for x in arcs])
    A, n, vote, med = align.fit(I, H)
    for l in L:
        for k in ("hinge", "open", "shut"):
            l["s_" + k] = align.apply(A, np.array(l[k])).tolist()
    return A, L, arcs, {"pairs": n, "median_pt": med}


def wall_ink(tag, z, seal_m=1.6):
    """Heavy strokes only (the clinic's wall pen is 0.79 pt), gaps along walls up to seal_m closed. Door chords are added by the caller.
    shortcut: picks walls by line weight, which works for this CAD set only; scans and other offices need a learned or Spark wall mask."""
    p = page(tag)
    H, W = int(p.rect.height * z) + 1, int(p.rect.width * z) + 1
    im = np.zeros((H, W), np.uint8)
    for d in p.get_drawings():
        if d["type"] != "s" or (d.get("width") or 0) < 0.7:
            continue
        for it in d["items"]:
            if it[0] == "l":
                cv2.line(im, (int(it[1].x * z), int(it[1].y * z)), (int(it[2].x * z), int(it[2].y * z)), 1, 2)
            elif it[0] == "re":
                r = it[1]
                cv2.rectangle(im, (int(r.x0 * z), int(r.y0 * z)), (int(r.x1 * z), int(r.y1 * z)), 1, 2)
    k = int(seal_m * PT_PER_M * z)
    return im | cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.ones((1, k), np.uint8)) | cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.ones((k, 1), np.uint8))


def outside(free):
    """Largest free region. free is 1 where no wall."""
    n, lab = cv2.connectedComponents(free, connectivity=4)
    return lab, 1 + int(np.bincount(lab[lab > 0] - 1).argmax())


def render(tag, dpi, clip=None):
    pix = page(tag).get_pixmap(dpi=dpi, clip=clip)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3][:, :, ::-1].copy()
