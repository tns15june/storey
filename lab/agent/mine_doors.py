"""Door tags and arcs on Capitol and Baxter sheets, from lab/modern code. Saves pairs with tag positions and contact crops. No API."""
import json, math, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/modern"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/agent")
import numpy as np, pymupdf as fitz, cv2
from scipy.optimize import linear_sum_assignment
import vec, measure as M, doors as D, draw
O = "/Users/tsharma/Github/Meta/data/lab/agent/mine/"
res = []
for id in ("mml_capitol", "mml_baxter"):
    for f in M.CAT[id]["floors"]:
        pg = vec.open_page(id, f["file"], f["page"])
        segs, _ = vec.segments(pg)
        rect, note, _ = vec.plan_panel(pg, segs, rule_layer="A-anno-htch-docs")
        panel = fitz.Rect(rect * pg.rotation_matrix)
        ipp = vec.inches_per_pt(note); sched = M.door_schedule(pg)
        arcs = D.door_swings(pg, panel, ipp); mk = D.marks(pg, panel, sched)
        C = np.array([[math.dist(a["mid"], (m[1], m[2])) for a in arcs] for m in mk])
        pairs = []
        for i, j in zip(*linear_sum_assignment(C)):
            pairs.append(dict(mark=mk[i][0], tag=[mk[i][1], mk[i][2]], arc=arcs[j]["mid"], c=arcs[j]["c"], r_in=arcs[j]["r_in"], d=round(float(C[i, j]), 1), sched=sched[mk[i][0]]))
        res.append(dict(id=id, level=str(f["level"]), file=f["file"], page=f["page"], rot=pg.rotation, panel=list(panel), ipp=ipp, marks=mk, arcs=arcs, pairs=pairs))
        print(id, f["level"], "rot", pg.rotation, "marks", len(mk), "arcs", len(arcs), "pairs", [(p["mark"], p["d"]) for p in pairs])
        c = panel
        im, P, s = draw.render(pg, c, long=2000)
        for m in mk:
            cv2.circle(im, P(m[1], m[2]), 16, (255, 0, 0), 2); cv2.putText(im, m[0], P(m[1] + 6, m[2]), 0, .6, (255, 0, 0), 2)
        for a in arcs:
            cv2.circle(im, P(*a["mid"]), 9, (0, 160, 0), 2)
        for p in pairs:
            cv2.line(im, P(*p["tag"]), P(*p["arc"]), (0, 0, 255) if p["d"] < 75 else (0, 200, 255), 2)
        cv2.imwrite(O + f"doors_{id}_{f['level']}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(res, open(O + "doors.json", "w"), default=float)
