"""Where seeds landed: every cluster's loose, snap and Spark-crop box on its seed floor over the truth boxes of that floor. No API."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import vis
for st in sys.argv[1:]:
    s = json.load(open(SPK / f"{st}_seeds.json")); cl = json.load(open(SPK / f"{st}_clusters.json"))
    fr = frames(st); T = TRUTH[st]
    tiles = []
    for i, c in enumerate(cl):
        f = c["seed_floor"]
        L = s["loose"][i][2]; S_ = s["snap"][i][2]; A = s["sparkcrop"][i][2]
        a = np.array([L, S_, A]); pad = 40
        R_ = (max(0, int(a[:, 0].min() - pad)), max(0, int(a[:, 1].min() - pad)), min(fr[0].shape[1], int(a[:, 2].max() + pad)), min(fr[0].shape[0], int(a[:, 3].max() + pad)))
        z = min(3.0, 300 / max(R_[2] - R_[0], R_[3] - R_[1]))
        ms = [(L, (0, 200, 255), f"{i+1} f{f} {c['floors']}"), (S_, (0, 170, 0), ""), (A, (0, 0, 255), "")]
        ms += [(x["boxes"][str(f)], (255, 0, 255), "") for x in T["stairs"].values() if str(f) in x["boxes"]]
        tiles.append(vis.grid_crop(fr[f], R_, z, 500, ms))
    W = 0; out = []; line = []
    for t in tiles:
        t = cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if W + t.shape[1] > 1900:
            out.append(line); line = []; W = 0
        line.append(t); W += t.shape[1]
    out.append(line)
    rows = []
    for l in out:
        h = max(t.shape[0] for t in l)
        v = np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in l])
        rows.append(cv2.copyMakeBorder(v, 0, 0, 0, max(0, 1900 - v.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))[:, :1900])
    p = vis.V / f"seeds_{st}.jpg"
    cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(p, len(cl))
