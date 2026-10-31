"""Second pass on habs_va_bldg1 and wfu_huffman: real plan boxes, and three motion models. Judged by how tightly the audit truth boxes stack after warping."""
import warnings; warnings.filterwarnings("ignore")
import json, sys
import cv2, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg, stacks as st
T = json.load(open(st.ROOT / "lab/audit/truth.json"))
SP = json.load(open(reg.OUT / "spark_sheets.json"))
order = {"habs_va_bldg1": [0, 1, 2, 3, 4]}  # audit f index -> align sheet index, both follow level order


def box_of(name, i, shape):
    h, w = shape[:2]
    if name == "wfu_huffman":
        return (101, 101, 1919, 2727)  # modern/stack.py rect in points, x 2.525: inside the sheet frame, above the title block
    b = SP[f"{name}/{order[name][i]}"]["read"]["plan_box"]
    p = int(0.02 * max(w, h))
    return (max(0, int(b["x1"] * w / 1000 - p)), max(0, int(b["y1"] * h / 1000 - p)), min(w, int(b["x2"] * w / 1000 + p)), min(h, int(b["y2"] * h / 1000 + p)))


def spread(name, Hs):
    """Per truth stair: max distance of warped box centres from their median, on floors it is drawn."""
    out = {}
    for k, s in T[name].items():
        c = []
        for f, b in s["boxes"].items():
            p = reg.apply(Hs[int(f)], [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]])[0]
            c.append(p)
        c = np.array(c)
        out[k] = round(float(np.linalg.norm(c - np.median(c, 0), axis=1).max()), 1)
    return out


for name, n in (("habs_va_bldg1", 5), ("wfu_huffman", 4)):
    ims = [cv2.imread(str(st.AUD / name / f"f{i}.png")) for i in range(n)]
    bx = [box_of(name, i, ims[i].shape) for i in range(n)]
    print(name, bx)
    best = None
    for model in ("similarity", "affine", "homography"):
        Hs, info = st.align_stack(ims, 1, bx, model)
        sp = spread(name, Hs)
        print(model, {k: v["inliers"] for k, v in info.items()}, sp, flush=True)
        if best is None or sum(sp.values()) < sum(best[1].values()):
            best = (model, sp, Hs, info)
    print("keep", best[0])
    st.save(name, ims, best[2], 1, best[3])
