"""Part 3: does SAM stack tracking need alignment? bme_f full factorial, the others one factor at a time around well/r4/2000.
Each run is cached by key, so a rerun spends nothing. Hit: mask centroid within 40 px of the stair (2000 px sheet) in reference coords."""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "lab")
sys.path.insert(0, "lab/align")
import metalab as m
import reg

SAM = reg.OUT / "sam"
SAM.mkdir(exist_ok=True)
Hb = json.load(open(reg.OUT / "H_blocks.json"))
TOL = 40
# ref floor, stair boxes on it (2000 px sheet coords), and (stair, floor) pairs seen to have no stair, by eye from truth/stairs_*.jpg
B = {
    "bme_f": (1, {1: (848, 900, 945, 1015), 2: (1175, 660, 1275, 740), 3: (512, 652, 612, 735), 4: (418, 215, 530, 280), 5: (1275, 228, 1385, 290)}, {(4, 3), (5, 3)}),
    "bme_k": (1, {1: (780, 689, 870, 779), 2: (1099, 689, 1189, 779), 3: (255, 674, 345, 764), 4: (1624, 683, 1714, 773), 5: (949, 230, 1039, 320)}, set()),
    "schenley": (1, {1: (964, 80, 1054, 170), 2: (191, 840, 291, 940), 3: (1706, 852, 1806, 952)}, set()),
    "wfu_efird": (1, {1: (766, 815, 856, 905)}, set()),
    "nibs_clinic": (0, {1: (845, 805, 935, 895), 2: (1042, 355, 1092, 425), 3: None}, {(1, 1)}),
    # held out: added only to validate the recipe
    "habs_stelizabeths_center": (1, {1: (308, 861, 398, 951), 2: (1624, 861, 1714, 951), 3: (865, 766, 945, 806)}, set()),
    "habs_stelizabeths_atkins": (1, {1: (680, 571, 816, 628)}, set()),
    "mml_capitol": (1, {1: (1352, 620, 1402, 770)}, {(1, 0)}),
}


def boxes_for(bid):
    ref, bx, absent = B[bid]
    if bid == "nibs_clinic":  # staff-entry stair sits just right of landmark corner13
        t = json.load(open(reg.OUT / "truth/truth.json"))["nibs_clinic"]["corner13"]["ref"]
        bx = dict(bx)
        bx[3] = (t[0] - 13, t[1] - 28, t[0] + 77, t[1] + 32)
    return ref, bx, absent


def stack_H(bid, tier, f, ref):
    if f == ref or tier == "none":
        return np.eye(3)
    if tier == "well":
        return np.array(Hb[f"{bid}/{f}"])
    if tier == "rough":
        return reg.coarse(reg.context(bid, f, ref))
    raise ValueError(tier)


def run(cfg):
    bid, tier, rep, res = cfg
    key = f"{bid}_{tier}_r{rep}_{res}"
    path = SAM / f"{key}.json"
    if path.exists():
        return json.load(open(path))
    ref, boxes, absent = boxes_for(bid)
    n = len(reg.INDEX[bid])
    ims = [reg.sheet(bid, i) for i in range(n)]
    H0, W0 = ims[ref].shape[:2]
    if tier == "none":
        H0, W0 = max(i.shape[0] for i in ims), max(i.shape[1] for i in ims)
    S = [stack_H(bid, tier, f, ref) for f in range(n)]
    s = res / max(H0, W0)
    w, h = int(W0 * s) & ~1, int(H0 * s) & ~1
    sx, sy = w / W0, h / H0
    vid = SAM / f"{key}.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "4", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(vid)], stdin=subprocess.PIPE)
    for f in range(n):
        fr = cv2.warpPerspective(ims[f], S[f], (W0, H0), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        for _ in range(rep):
            p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    pr = f"<{ref * rep}f>" + ",".join(f"{i}<|box;x1={int(b[0] * sx)};y1={int(b[1] * sy)};x2={int(b[2] * sx)};y2={int(b[3] * sy)};w={w};h={h}|>" for i, b in boxes.items())
    t0 = time.time()
    recs = m.sam_video(pr, vid, frames=n * rep, direction="both")
    dt = time.time() - t0
    mb = vid.stat().st_size / 1e6
    vid.unlink()
    out = {}
    for r in recs:
        if r.frame.frame_index % rep != rep - 1:
            continue
        f = r.frame.frame_index // rep
        mask = m.raster(r)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        c = np.array([(xs.mean() + r.bounds.left) / sx, (ys.mean() + r.bounds.top) / sy])  # stack px at 2000 scale
        q = reg.apply(np.array(Hb[f"{bid}/{f}"]) @ np.linalg.inv(S[f]), [c])[0]  # reference coords
        out[f"{r.object_id}/{f}"] = {"stack_xy": c.round(1).tolist(), "ref_xy": q.round(1).tolist(), "area": int(len(xs) / (sx * sy)), "conf": None if r.confidence is None else round(float(r.confidence), 3)}
    res_ = {"key": key, "bid": bid, "tier": tier, "rep": rep, "res": res, "frames": n * rep, "seconds": round(dt, 1), "video_mb": round(mb, 2), "w": w, "h": h, "masks": out, "records": len(recs)}
    json.dump(res_, open(path, "w"), indent=1)
    return res_


def score(r):
    ref, boxes, absent = boxes_for(r["bid"])
    n = len(reg.INDEX[r["bid"]])
    hit = miss = wrong = tn = fp = 0
    bad = []
    for i, b in boxes.items():
        ctr = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        for f in range(n):
            if f == ref:
                continue
            mk = r["masks"].get(f"{i}/{f}")
            if (i, f) in absent:
                if mk is None:
                    tn += 1
                else:
                    fp += 1; bad.append(f"fp s{i} f{f}")
            elif mk is None:
                miss += 1; bad.append(f"miss s{i} f{f}")
            elif np.linalg.norm(np.array(mk["ref_xy"]) - ctr) <= TOL:
                hit += 1
            else:
                wrong += 1; bad.append(f"wrong s{i} f{f} {np.linalg.norm(np.array(mk['ref_xy']) - ctr):.0f}px")
    t = hit + miss + wrong + tn + fp
    return dict(correct=hit + tn, total=t, hit=hit, miss=miss, wrong=wrong, tn=tn, fp=fp, bad=bad)


def validation():
    c = []
    for b in ("bme_f", "bme_k", "schenley", "wfu_efird", "nibs_clinic", "habs_stelizabeths_center", "habs_stelizabeths_atkins", "mml_capitol"):
        c += [(b, t, 1, 960) for t in ("none", "rough", "well")]
        c += [(b, "well", 1, 1440)]
    return c


def configs():
    c = [("bme_f", t, r, s) for t in ("none", "rough", "well") for r in (1, 2, 4) for s in (960, 1440, 2000)]
    for b in ("bme_k", "schenley", "wfu_efird", "nibs_clinic"):
        c += [(b, t, 4, 2000) for t in ("none", "rough", "well")]
        c += [(b, "well", r, 2000) for r in (1, 2)]
        c += [(b, "well", 4, s) for s in (960, 1440)]
    return c


if __name__ == "__main__":
    cs = list(dict.fromkeys(configs() + validation()))
    if len(sys.argv) > 1:
        cs = [c for c in cs if c[0] == sys.argv[1]][:int(sys.argv[2])]
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(run, cs):
            s = score(r)
            print(f"{r['key']:36s} {s['correct']:3d}/{s['total']:<3d} hit {s['hit']} tn {s['tn']} miss {s['miss']} wrong {s['wrong']} fp {s['fp']} {r['seconds']}s {r['video_mb']}MB {s['bad'][:5]}", flush=True)
    print("spent", round(m.spent("align"), 4))
