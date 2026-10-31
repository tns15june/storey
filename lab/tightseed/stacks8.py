"""habs_va_bldg1 back to reg.py's default similarity: the motion model must not be picked by truth-box spread. Same plan boxes as stacks2. No API."""
import warnings; warnings.filterwarnings("ignore")
import json, sys
import cv2, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg, stacks as st
SP = json.load(open(reg.OUT / "spark_sheets.json"))
name = "habs_va_bldg1"
ims = [cv2.imread(str(st.AUD / name / f"f{i}.png")) for i in range(5)]
def box_of(i):
    h, w = ims[i].shape[:2]; b = SP[f"{name}/{i}"]["read"]["plan_box"]; p = int(0.02 * max(w, h))
    return (max(0, int(b["x1"] * w / 1000 - p)), max(0, int(b["y1"] * h / 1000 - p)), min(w, int(b["x2"] * w / 1000 + p)), min(h, int(b["y2"] * h / 1000 + p)))
Hs, info = st.align_stack(ims, 1, [box_of(i) for i in range(5)], "similarity")
st.save(name, ims, Hs, 1, info)
