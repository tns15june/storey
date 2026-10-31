"""Stack saved crops into pages for reading. Usage: montage.py <out prefix> <crop ids...>. Each page holds up to 4 crops side by side in two columns."""
import sys, warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
C = "/Users/tsharma/Github/Meta/data/lab/declared/crops/"
pre, ids = sys.argv[1], sys.argv[2:]
for i in range(0, len(ids), 4):
    ims = []
    for k in ids[i:i + 4]:
        a = cv2.imread(C + k + ".png")
        s = min(1000 / a.shape[1], 980 / a.shape[0], 1)
        ims.append(cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    rows = []
    for j in range(0, len(ims), 2):
        pair = ims[j:j + 2]
        h = max(x.shape[0] for x in pair)
        rows.append(np.hstack([cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 1000 - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in pair] + ([np.full((h, 1000, 3), 255, np.uint8)] if len(pair) == 1 else [])))
    cv2.imwrite(f"{pre}_{i // 4}.png", np.vstack(rows))
