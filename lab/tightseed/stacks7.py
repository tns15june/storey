"""Abbot again: keep blocks for foundation and upper, place attic by matching the section bubble against the upper panel. No API."""
import warnings; warnings.filterwarnings("ignore")
import json, sys
import fitz, cv2, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg, stacks as st, stacks5
stacks5.abbot()
from stacks4 import page, P
pg = fitz.open(P / "mml_abbot/THE-ABBOT.pdf")[1]
z = 1.744
pan = {"foundation": (84, 872, 1231, 1695), "main": (1241, 872, 2386, 1695), "upper": (1241, 40, 2386, 862), "attic": (84, 40, 1231, 862)}
ims = [page(pg, z, clip=fitz.Rect(*pan[k])) for k in ("foundation", "main", "upper", "attic")]
h = min(i.shape[0] for i in ims); w = min(i.shape[1] for i in ims)
ims = [i[:h, :w] for i in ims]
d = json.load(open(st.OUT / "mml_abbot/H.json"))
Hs = {int(k): np.array(v) for k, v in d["H"].items()}; info = {int(k): v for k, v in d["info"].items()}
g = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in ims]
# bubble in the upper panel: find the circle by Hough near the panel's left, then match that patch in the attic panel
c = cv2.HoughCircles(cv2.GaussianBlur(g[2][300:600, 40:220], (0, 0), 1), cv2.HOUGH_GRADIENT, 1, 50, param1=120, param2=25, minRadius=14, maxRadius=30)
cx, cy, r = c[0][0]; cx += 40; cy += 300
print("upper bubble", cx, cy, r)
p = 60
t = g[2][int(cy) - p:int(cy) + p, int(cx) - p:int(cx) + p]
res = cv2.matchTemplate(g[3], t, cv2.TM_CCOEFF_NORMED)
_, v, _, loc = cv2.minMaxLoc(res)
ax, ay = loc[0] + p, loc[1] + p
print("attic bubble", ax, ay, "ncc", round(v, 3))
T = np.eye(3); T[0, 2] = cx - ax; T[1, 2] = cy - ay
Hs[3] = Hs[2] @ T; info[3] = dict(via="bubble match to upper", ncc=round(v, 3))
st.save("mml_abbot", ims, Hs, 1, info)
