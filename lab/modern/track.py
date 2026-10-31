"""SAM floor-stack tracker on clean CAD sheets, across framing, resolution, registration and erase controls.

Run: cd /Users/tsharma/Github/Meta && LAB_NAME=modern LAB_BUDGET=4 LAB_TOTAL=60 .venv/bin/python lab/modern/track.py
Needs lab/modern/render.py and lab/modern/stack.py first. Ground truth boxes were checked by eye on montages.
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/modern")
import cv2, numpy as np
import metalab as m
import stack as S

OUT = S.OUT
P = json.loads((OUT / "prep.json").read_text())
# Ground truth in the registered frame, rotated page points. None means the stair is not on that floor.
# Efird is not registered well by walls, so its boxes are per floor in its registered frame, loose by up to 40 pt.
GT = {
    "mml_capitol": dict(prompt="main", objs={
        1: dict(name="rear exterior", ground=[1942, 166, 2021, 346], main=[1942, 170, 2022, 350], upper=[1942, 172, 2021, 352]),
        2: dict(name="front exterior", ground=[1942, 1138, 2021, 1254], main=[1942, 1133, 2022, 1249], upper=[1942, 1130, 2021, 1246]),
        3: dict(name="interior", ground=None, main=[1744, 859, 1807, 985], upper=[1744, 748, 1807, 952])}),
    "mml_baxter": dict(prompt="main", objs={
        1: dict(name="left", main=[1791, 713, 1854, 839], upper=[1792, 603, 1854, 807]),
        2: dict(name="right", main=[1864, 713, 1928, 839], upper=[1865, 603, 1928, 807])}),
    "nibs_clinic": dict(prompt="1", objs={
        1: {"name": "1CS3", "1": [1196, 286, 1249, 399], "2": [1194, 280, 1250, 400]},
        2: {"name": "1AS1", "1": [1581, 548, 1631, 644], "2": [1579, 547, 1634, 645]},
        3: {"name": "1BS2", "1": [1604, 1104, 1707, 1159], "2": [1582, 1103, 1702, 1153]}}),
    "wfu_efird": dict(prompt="1", objs={
        1: {"name": "central", "0": [469, 503, 508, 546], "1": [472, 520, 506, 568], "2": [469, 488, 499, 531], "3": [478, 542, 505, 583]}}),
}
NEG = {"mml_capitol": ("upper", 3), "mml_baxter": ("upper", 2), "nibs_clinic": ("2", 1), "wfu_efird": ("2", 1)}
REP = 4


def frames_for(id, reg, frame):
    """Frames and a function from registered points to frame pixels. frame is 'sheet2000', 'panel1000', 'panel150', 'panel200'."""
    p = P[id]
    rect = p["rect"]
    dpi = 200 if frame == "panel200" else 150
    Z = dpi / 72
    shifts = {r["level"]: (r["dx_pt"], r["dy_pt"]) for r in p["regs"]}
    shifts = {str(k): v for k, v in shifts.items()}
    fs, maps = [], {}
    for lv in p["levels"]:
        im = cv2.imread(str(OUT / f"render/{id}_{lv}_{dpi}.png"))
        dx, dy = shifts[lv]
        if frame == "sheet2000":
            s = 2000 / im.shape[1]
            fs.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            maps[lv] = lambda b, dx=dx, dy=dy, s=s: [(b[0] - dx) * Z * s, (b[1] - dy) * Z * s, (b[2] - dx) * Z * s, (b[3] - dy) * Z * s]
            continue
        sx, sy = (dx, dy) if reg else (0, 0)
        A = np.float32([[1, 0, (sx - rect[0]) * Z], [0, 1, (sy - rect[1]) * Z]])
        f = cv2.warpAffine(im, A, (int((rect[2] - rect[0]) * Z), int((rect[3] - rect[1]) * Z)), borderValue=(255, 255, 255))
        s = 1.0
        if frame == "panel1000":
            s = 1000 / f.shape[1]
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        fs.append(f)
        maps[lv] = lambda b, dx=dx - sx, dy=dy - sy, s=s: [(b[0] - dx - rect[0]) * Z * s, (b[1] - dy - rect[1]) * Z * s, (b[2] - dx - rect[0]) * Z * s, (b[3] - dy - rect[1]) * Z * s]
    return fs, maps


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / (u + 1e-9)


def run(id, frame, reg, erase=None):
    gt = GT[id]
    fs, maps = frames_for(id, reg, frame)
    levels = P[id]["levels"]
    tag = f"{id}__{frame}__{'reg' if reg else 'raw'}" + (f"__erase{erase[0]}_{erase[1]}" if erase else "")
    objs = {o: dict(d) for o, d in gt["objs"].items()}
    if erase:
        lv, o = erase
        b = maps[lv](objs[o][lv])
        pad = 6 * (fs[0].shape[1] / 2000 if frame != "sheet2000" else 1)
        x0, y0, x1, y1 = int(b[0] - pad), int(b[1] - pad), int(b[2] + pad), int(b[3] + pad)
        fs[levels.index(lv)][y0:y1, x0:x1] = (255, 255, 255)
        objs[o][lv] = None
    path = OUT / f"videos/{tag}.mp4"
    path.parent.mkdir(exist_ok=True)
    w, h = S.write_mp4(fs, path, REP)
    plv = gt["prompt"]
    start = levels.index(plv) * REP
    toks = []
    for o, d in objs.items():
        b = [int(round(v)) for v in maps[plv](d[plv])]
        toks.append(f"{o}<|box;x1={b[0]};y1={b[1]};x2={b[2]};y2={b[3]};w={w};h={h}|>")
    prompt = f"<{start}f>" + ",".join(toks)
    t = time.time()
    recs = m.sam_video(prompt, path, len(levels) * REP)
    dt = time.time() - t
    last = {}
    for r in recs:
        last.setdefault((int(r.object_id), r.frame.frame_index // REP), []).append(r)  # object_id arrives as text
    rows, ovl = [], [f.copy() for f in fs]
    for o, d in objs.items():
        for i, lv in enumerate(levels):
            if lv == plv:
                continue
            cand = [r for r in last.get((o, i), []) if r.frame.frame_index % REP == REP - 1]
            r = cand[0] if cand else None
            gb = maps[lv](d[lv]) if d[lv] else None
            row = dict(obj=o, name=d["name"], level=lv, gt="present" if gb else "absent")
            if r is None:
                row.update(mask=False, ok=gb is None)
            else:
                b = r.bounds
                box = [b.left, b.top, b.right, b.bottom]
                area = int(m.raster(r).sum())
                row.update(mask=True, box=box, area=area, conf=None if r.confidence is None else round(r.confidence, 3))
                if gb:
                    row["iou"] = round(iou(box, gb), 3)
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    row["center_in_gt"] = bool(gb[0] <= cx <= gb[2] and gb[1] <= cy <= gb[3])
                    row["ok"] = row["iou"] >= 0.3 or row["center_in_gt"]
                else:
                    row["ok"] = False
                c = {1: (0, 0, 255), 2: (0, 160, 0), 3: (255, 0, 0)}.get(o, (0, 160, 255))
                roi = ovl[i][box[1]:box[3], box[0]:box[2]]
                ras = m.raster(r)[: roi.shape[0], : roi.shape[1]]
                roi[ras] = (0.45 * roi[ras] + 0.55 * np.array(c)).astype(np.uint8)
            if gb:
                cv2.rectangle(ovl[i], (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (0, 200, 255), 3)
            rows.append(row)
    ndir = OUT / "overlays"
    ndir.mkdir(exist_ok=True)
    tile = np.hstack([cv2.resize(f, None, fx=700 / f.shape[1], fy=700 / f.shape[1], interpolation=cv2.INTER_AREA) for f in ovl])
    cv2.imwrite(str(ndir / f"{tag}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 85])
    res = dict(tag=tag, set=id, frame=frame, reg=reg, erase=erase, w=w, h=h, seconds=round(dt, 1), masks=len(recs), rows=rows,
               pairs=len(rows), correct=sum(bool(r["ok"]) for r in rows), mp4_mb=round(path.stat().st_size / 1e6, 2))
    print(tag, f"{res['correct']}/{res['pairs']}", f"{dt:.0f}s", f"{w}x{h}", flush=True)
    return res


if __name__ == "__main__":
    jobs = []
    for id in GT:
        for frame, reg in (("sheet2000", False), ("panel1000", True), ("panel150", False), ("panel150", True), ("panel200", True)):
            jobs.append((id, frame, reg, None))
        jobs.append((id, "panel150", True, NEG[id]))
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda j: run(*j), jobs))
    (OUT / "track.json").write_text(json.dumps(res, indent=1))
    print("spent", m.spent("modern"))
