"""Overlay per stair per floor for one method: hand box magenta, seed blue, mask green when right, red when wrong, grey under conf .5. No API."""
import sys
sys.path.insert(0, "lab/tightseed")
from core import *
import vis


def view(st, me, stairs=None, zoom=None, pad=.6, out=None, maxw=1900):
    tr = json.load(open(SAMD / f"{st}__{me}.json")); T = TRUTH[st]; fr = frames(st)
    rows, _ = score(st, tr)
    own = {}
    for o, f, b in tr["seeds"]:
        for k in assign(st, b):
            own.setdefault(k, []).append((str(o), f, b))
    bars = []
    for k, s in T["stairs"].items():
        if s["cls"] == "amb" or (stairs and k not in stairs):
            continue
        a = np.array(list(s["boxes"].values()))
        w, h = a[:, 2].max() - a[:, 0].min(), a[:, 3].max() - a[:, 1].min()
        R_ = (max(0, int(a[:, 0].min() - pad * max(w, 60))), max(0, int(a[:, 1].min() - pad * max(h, 60))), min(fr[0].shape[1], int(a[:, 2].max() + pad * max(w, 60))), min(fr[0].shape[0], int(a[:, 3].max() + pad * max(h, 60))))
        z = zoom or min(3.0, 340 / max(R_[2] - R_[0], R_[3] - R_[1]))
        tiles = []
        for f in range(T["n"]):
            ms = []
            if str(f) in s["boxes"]:
                ms.append((s["boxes"][str(f)], (255, 0, 255), k))
            for o, sf, b in own.get(k, ()):
                if sf == f:
                    ms.append((b, (255, 120, 0), "seed"))
                mk = tr["tracks"].get(o, {}).get(str(f))
                if mk and mk["area"] > 0:
                    ok = has(mk); c = (0, 170, 0) if ok and str(f) in s["boxes"] else ((0, 0, 255) if ok else (150, 150, 150))
                    ms.append((mk["box"], c, f"{mk['conf']}"))
            t = vis.grid_crop(fr[f], R_, z, 500, ms)
            cv2.putText(t, f"f{f}" + ("" if str(f) in s["boxes"] else " ABSENT"), (4, t.shape[0] - 6), 0, .5, (0, 0, 255), 1)
            tiles.append(cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        bars.append(np.hstack(tiles))
    W = min(maxw, max(b.shape[1] for b in bars))
    bars = [cv2.copyMakeBorder(b, 0, 0, 0, W - b.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) if b.shape[1] < W else b[:, :W] for b in bars]
    p = vis.V / (out or f"h_{st}_{me}.jpg")
    cv2.imwrite(str(p), np.vstack(bars), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return p


if __name__ == "__main__":
    st, me = sys.argv[1], sys.argv[2]
    print(view(st, me, stairs=sys.argv[3].split(",") if len(sys.argv) > 3 else None))
