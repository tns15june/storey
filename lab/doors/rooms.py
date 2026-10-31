"""Doors per room: close door openings, label free space, count doors touching each room. Key: IfcRelSpaceBoundary. No API calls."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import ifc_truth as t, key as K
from exits_hybrid_lib import cluster

e = t.load()
door_ids = {i for i, v in e.items() if v[0] == "IFCDOOR"}
spaces = {}
for i, v in e.items():
    if v[0] == "IFCSPACE":
        spaces[i] = {"name": v[1][2].strip("'"), "long": v[1][7].strip("'"), "doors": set(), "M": t.world(e, t.ref(v[1][5]))}
for i, v in e.items():
    if v[0] == "IFCRELSPACEBOUNDARY" and v[1][5].strip() != "$":
        sp, el = t.ref(v[1][4]), t.ref(v[1][5])
        if el in door_ids and sp in spaces:
            spaces[sp]["doors"].add(el)


def points_under(i, seen):
    if i in seen or i not in e:
        return []
    seen.add(i)
    n, a = e[i]
    if n == "IFCCARTESIANPOINT":
        return [t.nums(a[0])]
    out = []
    for k in t.refs(",".join(a)):
        out += points_under(k, seen)
    return out


def centre(i):
    """Footprint centre of a space, world XY, from its representation points."""
    pts = [(p + [0])[:3] for p in points_under(t.ref(e[i][1][6]), set()) if len(p) >= 2]
    if not pts:
        return None
    P = np.array(pts)
    M = spaces[i]["M"]
    w = (M @ np.c_[P, np.ones(len(P))].T).T[:, :2]
    return ((w.min(0) + w.max(0)) / 2).tolist()


res = {}
for tag in K.SHEETS:
    A, L, arcs, reg = K.clinic_key(tag)
    doors = cluster([a for a in arcs if a["leaf"]], K.PT_PER_M)
    z = 110 / 72
    ink = K.wall_ink(tag, z)
    for a in doors:
        cv2.line(ink, (int(a["cx"] * z), int(a["cy"] * z)), (int(a["shut_end"][0] * z), int(a["shut_end"][1] * z)), 1, 5)
    n, lab = cv2.connectedComponents(1 - ink, connectivity=4)
    room_doors = defaultdict(set)
    for di, a in enumerate(doors):
        h, s = np.array([a["cx"], a["cy"]]), np.array(a["shut_end"])
        mid, u = (h + s) / 2, (s - h) / np.linalg.norm(s - h)
        nrm = np.array([-u[1], u[0]])
        for sgn in (1, -1):
            p = (mid + sgn * nrm * 0.5 * K.PT_PER_M) * z
            x, y = int(p[0]), int(p[1])
            w = lab[y - 3:y + 4, x - 3:x + 4]
            ids = w[w > 0]
            if ids.size:
                room_doors[int(np.bincount(ids).argmax())].add(di)
    ok = tot = skipped = 0
    near = 0
    per = Counter()
    for i, sp in spaces.items():
        if not sp["name"] or sp["name"][0] != ("1" if tag == "F1" else "2"):
            continue
        c = centre(i)
        if not c:
            skipped += 1
            continue
        xy = (A[:2, :2] @ np.array(c) + A[:2, 2]) * z
        x, y = int(xy[0]), int(xy[1])
        w = lab[y - 6:y + 7, x - 6:x + 7]
        ids = w[w > 0]
        if not ids.size:
            skipped += 1
            continue
        r = int(np.bincount(ids).argmax())
        tn, pn = len(sp["doors"]), len(room_doors.get(r, ()))
        tot += 1
        ok += min(tn, 2) == min(pn, 2)
        per[(min(tn, 2), min(pn, 2))] += 1
    res[tag] = {"spaces_scored": tot, "skipped": skipped, "one_vs_two_agree": ok / max(tot, 1), "confusion_truth_pred": {f"{a}->{b}": c for (a, b), c in sorted(per.items())}}
    print(tag, res[tag], flush=True)
(K.OUT / "rooms.json").write_text(json.dumps(res, indent=1))
