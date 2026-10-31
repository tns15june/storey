"""Every room to its nearest exit on Storey's walk mask; which routes cross a wall of the IFC truth mask? No API."""
import json, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/agent")
from routelib import *
O = "/Users/tsharma/Github/Meta/data/lab/agent/mine/"
res = {}
for fl in (1, 2):
    d = load(fl); rows = json.load(open(K.O + f"clinic_f{fl}_rows.json"), parse_constant=lambda c: float("inf"))
    # exits: the seeds of the earlier run, thinned, inside the walk mask
    seeds = [s for s in rows["seeds"][::40] if d["walk"][int(s[1]), int(s[0])]]
    rooms = rooms_of(d["lab"], d["walk"]); snap = snapper(d["walk"]); out = []
    mcp = MCP_Geometric(np.where(d["walk"], 1.0, -1.0), fully_connected=True)
    ends = list({snap(e) for e in seeds})
    cum, _ = mcp.find_costs(ends)  # one sweep from all exits
    for i, (x, y) in rooms.items():
        if not d["walk"][y, x]:
            sy, sx = snap((x, y))
        else:
            sy, sx = y, x
        if not np.isfinite(cum[sy, sx]):
            continue
        p = mcp.traceback((sy, sx))
        pts, t = taut(d["walk"], p)
        cr = crossings(pts, d["gt"])
        out.append(dict(room=i, xy=[x, y], m=round(t / K.PXM, 2), n=len(pts), cross=cr, pts=[[int(a), int(b)] for a, b in pts]))
    res[fl] = out
    print(fl, "rooms routed", len(out), "routes crossing truth", sum(bool(r["cross"]) for r in out))
    for r in sorted(out, key=lambda r: -sum(c[2] for c in r["cross"]))[:12]:
        print(" ", r["room"], r["xy"], r["m"], r["cross"])
json.dump(res, open(O + "routes.json", "w"))
