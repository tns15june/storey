"""Show audit truth boxes warped into tightseed frames, to check they sit on treads. No API."""
import sys, json
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import numpy as np, cv2, reg, vis
ROOT = vis.ROOT
AT = json.load(open(ROOT / "lab/audit/truth.json"))


def warp_box(b, H):
    p = reg.apply(H, [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
    return [round(float(v), 1) for v in (*p.min(0), *p.max(0))]


def warped(name):
    Hs = {int(k): np.array(v) for k, v in json.load(open(vis.S / name / "H.json"))["H"].items()}
    return {k: {"cls": s["cls"], "boxes": {f: warp_box(b, Hs[int(f)]) for f, b in s["boxes"].items()}} for k, s in AT[name].items()}


def show(name, keys, tag, pad=45, zoom=3):
    t = warped(name)
    fr = vis.frames(name); rows = []
    for k in keys:
        bs = t[k]["boxes"]
        a = np.array(list(bs.values()))
        R = (int(a[:, 0].min() - pad), int(a[:, 1].min() - pad), int(a[:, 2].max() + pad), int(a[:, 3].max() + pad))
        ts = []
        for i, im in enumerate(fr):
            m = [(bs[str(i)], (0, 200, 0), k)] if str(i) in bs else []
            ts.append(vis.grid_crop(im, R, zoom, 10, m))
        rows.append(np.hstack([cv2.copyMakeBorder(x, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 255)) for x in ts]))
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    p = vis.V / f"tv_{name}_{tag}.jpg"
    cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(p)


if __name__ == "__main__":
    show("bme_k", ["N_center", "W_grand", "E_grand"], "a", zoom=2)
    show("bme_k", ["W_wing", "E_wing", "spiral_NE", "curve_SW"], "b", zoom=2.5)
    show("bme_k", ["svc_NW", "svc_SW", "svc_NE", "svc_SE"], "c", zoom=3)
    show("habs_va_bldg1", ["west", "east", "central"], "a", pad=30, zoom=2)
    show("wfu_huffman", ["central", "ext_E"], "a", pad=30, zoom=1.5)
    show("wfu_huffman", ["south"], "b", pad=30, zoom=1)
