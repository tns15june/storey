"""Render the other frozen mis-pairs through the agent's own view tool. No API."""
import sys
sys.path.insert(0, "lab"); sys.path.insert(0, "lab/agent")
import cv2, numpy as np
import fixture as F
t = F.Tools()
tiles = []
for rid in ("D08", "D31", "D33", "D37"):
    r = next(x for x in t.dims if x["id"] == rid)
    c = r["crop"]
    n, im = t.view_sheet(dict(sheet="055", x0=c[0], y0=c[1], x1=c[2], y1=c[3], overlay_row=rid))
    print(rid, r["printed_mm"], r["measured_mm"], c)
    z = min(560 / im.shape[1], 560 / im.shape[0])
    q = cv2.resize(im, None, fx=z, fy=z, interpolation=cv2.INTER_AREA)
    q = cv2.copyMakeBorder(q, 18, 580 - q.shape[0] - 18 if q.shape[0] < 562 else 0, 0, 580 - q.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(q, f"{rid} printed {r['printed_mm']} measured {r['measured_mm']}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, .42, (200, 0, 0), 1, cv2.LINE_AA)
    tiles.append(cv2.copyMakeBorder(q, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
cv2.imwrite("data/lab/audit2/agent_mispairs.jpg", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 90])
