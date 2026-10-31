"""Post hoc, no API, used by no score. Does the floor that failed the gate pass against its neighbour instead of the reference floor?"""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/presence")
import numpy as np
import pres
from pres import T, reg, ctr, apply_pt
pres.load_spark()
b = "wfu_huffman"
c = reg.context(b, 3, 2)
H32 = reg.blocks(c)
print("f3 to f2 inliers", c.get("inliers"), "of", c.get("matches"), "scale", round(float(np.sqrt(abs(np.linalg.det(H32[:2, :2])))), 4))
A = pres.alignment(b)
H = np.array(A["2"]["H"]) @ H32
v = T[b]["objects"]["central"]["boxes"]
print("central stair f3 chained to ref, error px", round(float(np.linalg.norm(apply_pt(H, ctr(v["3"])) - np.array(ctr(v["1"])))), 1))
