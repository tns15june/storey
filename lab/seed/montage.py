"""One image per building: a row per cluster, a column per floor, SAM mask painted, Spark member boxes drawn. Usage: montage.py variant [buildings]"""
import sys, glob; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed")
from common import *
var = sys.argv[1]
for p in sorted(glob.glob(str(OUT / "sam" / f"result_{var}__*.json"))):
    bid = Path(p).stem.split("__")[1]
    if sys.argv[2:] and bid not in sys.argv[2:]:
        continue
    res = json.load(open(p))
    run = json.load(open(OUT / "sam" / f"union_{var}__{bid}.json"))
    masks = np.load(OUT / "sam" / f"union_{var}__{bid}.npz")
    nf = len(CAT[bid]["floors"])
    fr = [cv2.imread(str(OUT / bid / f"f{i}.png")) for i in range(nf)]
    owner = {r.get("cluster"): tid for tid, r in res["summary"]["raw"]["rows"].items() if r.get("seeded")}
    rows = []
    for i, c in enumerate(res["clusters"]):
        b = c["seed_box"]
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        r = max(70, max(b[2] - b[0], b[3] - b[1]) // 2 + 50)
        tiles = []
        for f in range(nf):
            x1, y1, x2, y2 = max(0, cx - r), max(0, cy - r), min(fr[f].shape[1], cx + r), min(fr[f].shape[0], cy + r)
            t = fr[f].copy()
            k = f"{i + 1}_{f}"
            mk = run["tracks"].get(str(i + 1), {}).get(str(f))
            if k in masks and mk and "box" in mk:
                bb = mk["box"]; ras = masks[k]
                roi = t[bb[1]:bb[1] + ras.shape[0], bb[0]:bb[0] + ras.shape[1]]
                ras = ras[:roi.shape[0], :roi.shape[1]]
                col = (0, 200, 0) if (mk.get("conf") or 0) >= 0.5 else (0, 160, 255)
                roi[ras] = (roi[ras] * 0.5 + np.array(col) * 0.5).astype(np.uint8)
            for ff, bx in c["members"]:
                if ff == f:
                    cv2.rectangle(t, tuple(bx[:2]), tuple(bx[2:]), (255, 0, 255), 1)
            s = 220 / max(x2 - x1, y2 - y1)
            t = cv2.resize(t[y1:y2, x1:x2], (220, 220), interpolation=cv2.INTER_AREA)
            lab = f"f{f} " + (f"{mk.get('conf')}" if mk and "box" in mk else "none")
            cv2.putText(t, lab, (3, 14), 0, 0.45, (0, 0, 255), 1)
            tiles.append(t)
        head = np.full((220, 150, 3), 255, np.uint8)
        g = res["gate"].get(str(i), {})
        for j, txt in enumerate([f"#{i + 1} {owner.get(i, 'FALSE?')}", f"seed f{c['seed_floor']}", f"spark {c['floors']}", "gate " + str(g.get("is_stair")), str(g.get("what", ""))[:20], str(g.get("what", ""))[20:40]]):
            cv2.putText(head, txt, (3, 18 + 20 * j), 0, 0.42, (0, 0, 0), 1)
        rows.append(np.hstack([head] + tiles))
    for pg in range(0, len(rows), 7):
        cv2.imwrite(str(OUT / f"montage_{var}_{bid}_{pg // 7}.jpg"), np.vstack(rows[pg:pg + 7]), [cv2.IMWRITE_JPEG_QUALITY, 85])
