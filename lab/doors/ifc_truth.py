"""Door answer key from the clinic IFC (no ifcopenshell): world XY, width, storey, IsExternal, swing axes."""
import json, math, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IFC = ROOT / "data/plans/nibs_clinic/Clinic_A_20110906_optimized.ifc"
OUT = ROOT / "data/lab/doors/clinic_ifc_doors.json"


def split(s):
    """Split a STEP argument list on top-level commas."""
    out, depth, cur, q = [], 0, "", False
    for ch in s:
        if ch == "'":
            q = not q
        if not q:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                out.append(cur); cur = ""; continue
        cur += ch
    out.append(cur)
    return out


def load():
    ents = {}
    for m in re.finditer(r"^#(\d+)=([A-Z0-9]+)\((.*)\);\s*$", IFC.read_text(errors="replace"), re.M):
        ents[int(m[1])] = (m[2], split(m[3]))
    return ents


def ref(s):
    return int(s.strip().lstrip("#"))


def refs(s):
    return [int(x) for x in re.findall(r"#(\d+)", s)]


def nums(s):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:E[-+]?\d+)?", s)]


def axis2(e, i):
    _, a = e[i]
    loc = nums(e[ref(a[0])][1][0])
    loc = (loc + [0, 0, 0])[:3]
    z = nums(e[ref(a[1])][1][0]) if a[1].strip() != "$" else [0, 0, 1]
    x = nums(e[ref(a[2])][1][0]) if a[2].strip() != "$" else [1, 0, 0]
    z, x = np.array(z, float) / np.linalg.norm(z), np.array(x, float) / np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, loc
    return M


def world(e, i):
    _, a = e[i]
    M = axis2(e, ref(a[1]))
    return world(e, ref(a[0])) @ M if a[0].strip() != "$" else M


def mapped_op(e, i):
    """IFCCARTESIANTRANSFORMATIONOPERATOR3D as a 4x4."""
    n, a = e[i]
    x = nums(e[ref(a[0])][1][0]) if a[0].strip() != "$" else [1, 0, 0]
    y = nums(e[ref(a[1])][1][0]) if a[1].strip() != "$" else [0, 1, 0]
    o = (nums(e[ref(a[2])][1][0]) + [0, 0, 0])[:3]
    s = float(a[3]) if a[3].strip() != "$" else 1.0
    z = nums(e[ref(a[4])][1][0]) if len(a) > 4 and a[4].strip() != "$" else [0, 0, 1]
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = np.array(x) * s, np.array(y) * s, np.array(z) * s, o
    return M


def leaves(e, d, W):
    """Swing arcs from the Plan representation: hinge, both arc ends and the leaf-tip end, in world metres."""
    out = []
    for q in refs(e[ref(d[6])][1][2]):
        if "Plan" not in e[q][1][1]:
            continue
        for it in refs(e[q][1][3]):
            if e[it][0] != "IFCMAPPEDITEM":
                continue
            rm = e[ref(e[it][1][0])][1]
            sr = e[ref(rm[1])][1]
            T = W @ mapped_op(e, ref(e[it][1][1]))
            for gs in refs(sr[3]):
                items = refs(e[gs][1][0]) if e[gs][0] == "IFCGEOMETRICSET" else [gs]
                lines = []
                for k in items:
                    if e[k][0] == "IFCPOLYLINE":
                        lines += [nums(e[p_][1][0])[:2] for p_ in refs(e[k][1][0])]
                for k in items:
                    if e[k][0] != "IFCTRIMMEDCURVE" or e[ref(e[k][1][0])][0] != "IFCCIRCLE":
                        continue
                    c = e[ref(e[k][1][0])][1]
                    pl = e[ref(c[0])][1]
                    o = np.array(nums(e[ref(pl[0])][1][0])[:2])
                    xd = np.array(nums(e[ref(pl[1])][1][0])[:2]) if pl[1].strip() != "$" else np.array([1.0, 0])
                    yd = np.array([-xd[1], xd[0]])
                    r = float(c[1])
                    t0, t1 = nums(e[k][1][1])[0], nums(e[k][1][2])[0]
                    pt = lambda t: o + r * (math.cos(math.radians(t)) * xd + math.sin(math.radians(t)) * yd)
                    a_, b_ = pt(t0), pt(t1)
                    dist = lambda v: min(np.linalg.norm(np.array(l) - v) for l in lines) if lines else 9
                    # the leaf rectangle touches the open arc end
                    op, sh = (a_, b_) if dist(a_) < dist(b_) else (b_, a_)
                    w3 = lambda v: (T @ np.array([v[0], v[1], 0, 1]))[:2].tolist()
                    out.append({"hinge": w3(o), "open": w3(op), "shut": w3(sh), "r": r * float(np.linalg.norm(T[:3, 0])), "sweep": abs(t1 - t0)})
    return out


def main():
    e = load()
    storey = {}
    for i, (t, a) in e.items():
        if t == "IFCRELCONTAINEDINSPATIALSTRUCTURE":
            for k in refs(a[4]):
                storey[k] = e[ref(a[5])][1][2].strip("'")
    psets = {}
    for i, (t, a) in e.items():
        if t == "IFCRELDEFINESBYPROPERTIES":
            d = e.get(ref(a[5]))
            if d and d[0] == "IFCPROPERTYSET":
                for k in refs(a[4]):
                    psets.setdefault(k, []).append(ref(a[5]))
    doors = []
    for i, (t, a) in e.items():
        if t != "IFCDOOR":
            continue
        M = world(e, ref(a[5]))
        ext = None
        for ps in psets.get(i, []):
            for p in refs(e[ps][1][4]):
                n = e.get(p)
                if n and n[0] == "IFCPROPERTYSINGLEVALUE" and n[1][0].strip("'") == "IsExternal":
                    ext = ".T." in n[1][2]
        doors.append({"id": i, "name": a[2].strip("'"), "x": M[0, 3], "y": M[1, 3], "z": M[2, 3],
                      "ax": [M[0, 0], M[1, 0]], "ay": [M[0, 1], M[1, 1]], "width": float(a[9]) if a[9].strip() != "$" else None,
                      "storey": storey.get(i), "external": ext, "leaves": leaves(e, a, M)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doors, indent=0))
    import collections
    print(len(doors), collections.Counter((d["storey"], d["external"]) for d in doors))
    print(collections.Counter(d["name"].split(":")[0] for d in doors).most_common(20))
    print("leaves", collections.Counter(len(d["leaves"]) for d in doors))
    return doors


if __name__ == "__main__":
    main()
