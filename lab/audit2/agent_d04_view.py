"""Render D04 and two neighbouring rows through the agent's own view tool, into audit2. No API."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/agent")
import cv2
import fixture as F
cls = [v for k, v in vars(F).items() if isinstance(v, type) and hasattr(v, "view_sheet")][0]
t = cls.__new__(cls)
try:
    t.__init__()
except TypeError:
    t.__init__(True)
for rid, c in (("D04", [655, 311, 733, 406]), ("D08", [756, 394, 820, 487])):
    n, im = t.view_sheet(dict(sheet="055", x0=c[0], y0=c[1], x1=c[2], y1=c[3], overlay_row=rid))
    print(n, im.shape)
    r = next(x for x in t.dims if x["id"] == rid)
    print({k: v for k, v in r.items()})
    cv2.imwrite(f"/Users/tsharma/Github/Meta/data/lab/audit2/agent_{rid}.png", im)
