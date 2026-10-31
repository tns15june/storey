"""Every new truth box on its frame, to catch label slips before the first model call. No API."""
import sys, json
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
import numpy as np, cv2, vis
T = json.load(open(vis.ROOT / "data/lab/tightseed/truth.json"))
for name, zoom in (("queens_royce1", 3), ("leonard4", 2.5)):
    fr = vis.frames(name); rows = []
    for k, s in T[name]["stairs"].items():
        a = np.array(list(s["boxes"].values()))
        R = [int(a[:, 0].min() - 18), int(a[:, 1].min() - 18), int(a[:, 2].max() + 18), int(a[:, 3].max() + 18)]
        ts = []
        for i, im in enumerate(fr):
            b = s["boxes"].get(str(i))
            ts.append(vis.grid_crop(im, R, zoom, 50, [(b, (0, 0, 255), k)] if b else []))
        rows.append(np.hstack([cv2.copyMakeBorder(x, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 160, 0)) for x in ts]))
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv2.imwrite(str(vis.V / f"tc_{name}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(vis.V / f"tc_{name}.jpg")
