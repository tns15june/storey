"""Two tracker tricks scored on CubiCasa truth: a camera zoom as a video, and a room followed through a revision."""
import json

import cv2
import numpy as np

import core as c
from data import OUT, load, poly_mask, rep_point, select

T = select()
data = {p: load(p) for p in T["plans"]}
ROOMS = [r for r in T["rooms"] if r["walled"]][::4][:6]
MORE = [r for r in T["rooms"] if r["walled"] and r not in ROOMS][1::4][:10]  # more rooms for the one case that looked good
SZ = 1024
N = 16


def bbox(q):
    return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()


def cam(im, q):
    """Affine matrices from the whole plan to a tight room view, on a log-zoom path. Pads with white."""
    h, w = im.shape[:2]
    x1, y1, x2, y2 = bbox(q)
    s0 = SZ / max(w, h)
    s1 = SZ / (max(x2 - x1, y2 - y1) * 1.6)
    c0, c1 = np.array([w / 2, h / 2]), np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    out = []
    for t in np.linspace(0, 1, N):
        s = s0 * (s1 / s0) ** t
        k = (s - s0) / (s1 - s0)
        ct = c0 + (c1 - c0) * k
        out.append(np.array([[s, 0, SZ / 2 - s * ct[0]], [0, s, SZ / 2 - s * ct[1]]]))
    return out


def warp(p, A):
    return p @ A[:, :2].T + A[:, 2]


def zoom(r, kind, start):
    im, g = data[r["pid"]]
    q = g["rooms"][r["i"]][1]
    As = cam(im, q)
    fr = [cv2.warpAffine(im, A, (SZ, SZ), flags=cv2.INTER_AREA if A[0, 0] < 1 else cv2.INTER_CUBIC, borderValue=(255, 255, 255)) for A in As]
    path = OUT / f"zoom_{r['pid']}_{r['i']}.mp4"
    c.encode(fr, path, repeat=1)
    fi = 0 if start == "overview" else N - 1
    q2 = warp(q, As[fi])
    b = bbox(q2)
    o = dict(box=(max(b[0], 0), max(b[1], 0), min(b[2], SZ - 1), min(b[3], SZ - 1))) if kind == "BOX" else dict(pos=[rep_point(q2)])
    res = c.video(c.obj_prompt([o], SZ, SZ, frame=fi), path, N, direction="forward" if fi == 0 else "backward")
    def score(f):
        gt = poly_mask(warp(q, As[f]), (SZ, SZ))
        rec = res.get((f, 1))
        mk = c.paste(rec[0], rec[2], (SZ, SZ)) if rec else np.zeros(gt.shape, bool)
        return c.iou(c.fill_holes(mk), gt) if mk.any() else 0.0, mk, gt, fr[f]
    a, m0, g0, f0 = score(0)
    z, m1, g1, f1 = score(N - 1)
    return dict(pid=r["pid"], i=r["i"], kind=kind, start=start, iou_overview=a, iou_detail=z), (c.overlay(f0, m0, g0), c.overlay(f1, m1, g1))


def revision(pid):
    im, g = data[pid]
    rs = [r for r in T["rooms"] if r["pid"] == pid]
    h, w = im.shape[:2]
    h -= h % 2
    w -= w % 2
    A = im[:h, :w].copy()
    shift = np.float32([[1, 0, 15], [0, 1, 10]])
    A2 = cv2.warpAffine(A, shift, (w, h), borderValue=(255, 255, 255))
    B = A.copy()
    truth = {}
    for k, r in enumerate(rs):
        q = g["rooms"][r["i"]][1]
        gt = poly_mask(q, im.shape)[:h, :w]
        x1, y1, x2, y2 = (int(v) for v in bbox(q))
        if k % 2 == 0 and x2 - x1 > 60:  # a new partition at 60 percent of the width
            px = x1 + int(0.6 * (x2 - x1))
            B[y1:y2, px - 3:px + 3] = 0
            left = gt.copy()
            left[:, px - 3:] = False
            truth[k] = left.sum() / gt.sum()
        else:
            truth[k] = 1.0
    B2 = cv2.warpAffine(B, shift, (w, h), borderValue=(255, 255, 255))
    path = OUT / f"rev_{pid}.mp4"
    c.encode([A, A2, B2], path, repeat=4)
    objs = []
    for r in rs:
        x1, y1, x2, y2 = bbox(g["rooms"][r["i"]][1])
        objs.append(dict(box=(max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1))))
    res = c.video(c.obj_prompt(objs, w, h), path, 12, direction="forward")
    rows = []
    for k, r in enumerate(rs):
        a = res.get((7, k + 1))
        b = res.get((11, k + 1))
        area = lambda x: float(c.fill_holes(x[2]).sum()) if x else 0.0
        rows.append(dict(pid=pid, k=k, truth=float(truth[k]), edited=bool(truth[k] < 1), ratio=area(b) / area(a) if area(a) else None, area_a=area(a)))
    return rows


if __name__ == "__main__":
    jobs = [(r, k, s) for r in ROOMS for k in ("BOX", "P") for s in ("overview", "detail")]
    jobs += [(r, "P", "detail") for r in MORE]

    def safe(j):
        try:
            return zoom(*j)
        except Exception as e:  # the tracker and the upload both returned server errors during this run
            print("zoom failed", j[0]["pid"], j[1], j[2], str(e)[:60], flush=True)

    out = [o for o in c.pmap(safe, jobs, n=4) if o]
    rows = [o[0] for o in out]
    (OUT / "zoom.json").write_text(json.dumps(rows))
    same = [o for o in out if o[0]["kind"] == "P" and o[0]["start"] == "detail"]

    def still(o):  # same tight frame, same point, one image call: is the tracker doing the work, or the framing
        r = next(x for x in T["rooms"] if x["pid"] == o[0]["pid"] and x["i"] == o[0]["i"])
        im, g = data[r["pid"]]
        q = g["rooms"][r["i"]][1]
        A = cam(im, q)[-1]
        f = cv2.warpAffine(im, A, (SZ, SZ), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        q2 = warp(q, A)
        res = c.sam(c.obj_prompt([dict(pos=[rep_point(q2)])], SZ, SZ), f)
        gt = poly_mask(q2, (SZ, SZ))
        return c.iou(c.fill_holes(res[0][3]), gt) if res else 0.0

    st = c.pmap(still, same, n=4)
    print("point on tight frame: tracker median detail", round(float(np.median([o[0]["iou_detail"] for o in same])), 3), "overview", round(float(np.median([o[0]["iou_overview"] for o in same])), 3),
          "n", len(same), ">=.7 overview", np.mean([o[0]["iou_overview"] >= .7 for o in same]), "| image call on the same frame median", round(float(np.median(st)), 3), ">=.7", np.mean(np.array(st) >= .7))
    for k in ("BOX", "P"):
        for s in ("overview", "detail"):
            v = [x for x in rows if x["kind"] == k and x["start"] == s]
            print("zoom", k, "prompt on", s, "n", len(v), "median IoU overview", round(float(np.median([x["iou_overview"] for x in v])), 3), "detail", round(float(np.median([x["iou_detail"] for x in v])), 3),
                  "overview>=.7", np.mean([x["iou_overview"] >= .7 for x in v]), "detail>=.7", np.mean([x["iou_detail"] >= .7 for x in v]))
    c.sheet([(o[1][0], f'{o[0]["kind"]} from {o[0]["start"]} ov {o[0]["iou_overview"]:.2f}') for o in out][:8] + [(o[1][1], f'{o[0]["kind"]} from {o[0]["start"]} det {o[0]["iou_detail"]:.2f}') for o in out][:8], OUT / "zoom_sheet.jpg")

    def safe_rev(p):
        try:
            return revision(p)
        except Exception as e:
            print("revision failed", p, str(e)[:60], flush=True)
            return []

    rev = sum(c.pmap(safe_rev, T["plans"], n=4), [])
    (OUT / "revision.json").write_text(json.dumps(rev))
    ed = [r for r in rev if r["edited"] and r["ratio"] is not None]
    un = [r for r in rev if not r["edited"] and r["ratio"] is not None]
    print("revision edited n", len(ed), "median |ratio-truth|", round(float(np.median([abs(r["ratio"] - r["truth"]) for r in ed])), 3), "share within .1", np.mean([abs(r["ratio"] - r["truth"]) <= .1 for r in ed]), "median ratio", round(float(np.median([r["ratio"] for r in ed])), 3), "median truth", round(float(np.median([r["truth"] for r in ed])), 3))
    print("revision control n", len(un), "median ratio", round(float(np.median([r["ratio"] for r in un])), 3), "share within .1 of 1", np.mean([abs(r["ratio"] - 1) <= .1 for r in un]))
    print("spent", round(c.m.spent("samcraft"), 3))
