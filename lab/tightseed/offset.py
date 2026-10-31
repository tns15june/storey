"""Is Spark's box error a fixed offset or scale? Per box, centre error to the nearest truth stair on that floor. No API."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
rows = {}
for st in STACKS:
    T = TRUTH[st]
    x1, y1, x2, y2 = T["crop"]; W, H = x2 - x1, y2 - y1
    for f in range(T["n"]):
        p = SPK / f"{st}_f{f}.json"
        if not p.exists():
            continue
        tb = [(k, s["boxes"][str(f)]) for k, s in T["stairs"].items() if str(f) in s["boxes"] and s["cls"] != "amb"]
        for b in json.load(open(p))["boxes"]:
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if not tb:
                continue
            k, t = min(tb, key=lambda q: np.hypot((q[1][0] + q[1][2]) / 2 - c[0], (q[1][1] + q[1][3]) / 2 - c[1]))
            d = ((t[0] + t[2]) / 2 - c[0], (t[1] + t[3]) / 2 - c[1])
            if np.hypot(*d) < 130:
                rows.setdefault(st, []).append((round((c[0] - x1) / W, 3), round((c[1] - y1) / H, 3), round(d[0], 1), round(d[1], 1), round((b[2] - b[0]) / max(1, t[2] - t[0]), 2), round((b[3] - b[1]) / max(1, t[3] - t[1]), 2)))
for st, r in rows.items():
    a = np.array(r)
    print(st, "n", len(a), "median dx dy", np.median(a[:, 2]).round(1), np.median(a[:, 3]).round(1), "abs med", np.median(np.abs(a[:, 2])).round(1), np.median(np.abs(a[:, 3])).round(1), "size ratio w h", np.median(a[:, 4]).round(2), np.median(a[:, 5]).round(2))
    if len(a) >= 4:
        A = np.c_[a[:, 0], np.ones(len(a))]
        cx = np.linalg.lstsq(A, a[:, 2], rcond=None)[0]; cy = np.linalg.lstsq(np.c_[a[:, 1], np.ones(len(a))], a[:, 3], rcond=None)[0]
        print("     fit dx = %.1f*u + %.1f   dy = %.1f*v + %.1f" % (cx[0], cx[1], cy[0], cy[1]))
        print("     ", [(x[0], x[1], x[2], x[3]) for x in r][:10])
