"""Presence gallery: each candidate stair at its building position on every floor, from the blocks fit, for an eye call."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg

H = json.load(open(reg.OUT / "H_blocks.json"))
# ref floor index, stair centres on the ref floor in 2000 px sheet coords
STAIRS = {
    "bme_f": (1, {"s0": (474, 247), "s1": (1330, 259), "s2": (562, 693), "s3": (1225, 700), "s4": (896, 957)}),
    "bme_k": (1, {"s0": (825, 734), "s1": (1144, 734), "s2": (300, 719), "s3": (1669, 728), "s4": (994, 275)}),
    "schenley": (1, {"s0": (1009, 125), "s1": (284, 922), "s2": (1753, 922)}),
    "wfu_efird": (1, {"s0": (811, 860)}),
    "nibs_clinic": (0, {"s0": (890, 850), "s1": (1078, 390), "s2": (1160, 882)}),
}
if __name__ == "__main__":
    for bid, (ref, st) in STAIRS.items():
        n = len(reg.INDEX[bid])
        rows = []
        for k, p in st.items():
            ts = []
            for f in range(n):
                q = reg.apply(np.linalg.inv(np.array(H[f"{bid}/{f}"])), [p])[0].astype(int)
                im = cv2.copyMakeBorder(reg.sheet(bid, f), 120, 120, 120, 120, cv2.BORDER_CONSTANT, value=(180, 180, 180))
                t = cv2.resize(im[q[1]:q[1] + 240, q[0]:q[0] + 240], (260, 260), interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(t, (85, 85), (175, 175), (0, 0, 255), 1)
                cv2.putText(t, f"{k} f{f}" + (" REF" if f == ref else ""), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 1, cv2.LINE_AA)
                ts.append(cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
            rows.append(np.hstack(ts))
        cv2.imwrite(str(reg.OUT / f"truth/stairs_{bid}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
