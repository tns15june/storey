"""wfu_huffman f3 failed against f1 (57 of 94 inliers, passed the gate, overlay wrong). Chain it through f2, the floor with the same south wing."""
import warnings; warnings.filterwarnings("ignore")
import json, sys
import cv2, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg, stacks as st
T = json.load(open(st.ROOT / "lab/audit/truth.json"))
name = "wfu_huffman"; R = (101, 101, 1919, 2727)
ims = [cv2.imread(str(st.AUD / name / f"f{i}.png")) for i in range(4)]
d = json.load(open(st.OUT / name / "H.json")); Hs = {int(k): np.array(v) for k, v in d["H"].items()}; info = {int(k): v for k, v in d["info"].items()}
best = None
for sc in (np.linspace(.84, 1.2, 19), np.linspace(.7, 1.0, 31)):
    c = st.ctx(ims[3], ims[2], R, R)
    H0 = reg.coarse(c, scales=sc)
    H = reg.blocks(c, H0=H0)
    print("f3->f2 inliers", c.get("inliers"), "of", c.get("matches"), "scale", round(float(np.hypot(H[0, 0], H[1, 0])), 4), H.round(3).tolist())
    if best is None or (c.get("inliers") or 0) > best[0]:
        best = (c.get("inliers") or 0, H, c.get("matches"))
Hs[3] = Hs[2] @ best[1]; info[3] = dict(inliers=best[0], matches=best[2], via="f2")
for k, s in T[name].items():
    c = np.array([reg.apply(Hs[int(f)], [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]])[0] for f, b in s["boxes"].items()])
    print(k, c.round(0).tolist())
st.save(name, ims, Hs, 1, info)
