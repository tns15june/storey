"""Dev only: e2e tracks for snap and Spark crop seeds at 960, plus an overlay of the seed boxes on main stairs."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import dev1, seeds as sd, tread, vis

CHECK = dict(steps=5, q90=.4)


def make_check(st, tr, fr=None):
    gr = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (fr or frames(st))]
    cache = {}
    def chk(o, f, mk):
        k = (o, f)
        if k not in cache:
            r = tread.score_box(gr[f], mk["box"])
            cache[k] = r["steps"] >= CHECK["steps"] and r["q90"] >= CHECK["q90"]
        return cache[k]
    return chk


if __name__ == "__main__":
    tiles = []
    for st in DEV:
        cl = dev1.loose_clusters(st); fr = frames(st)
        for name, method, kw in (("loose", "loose", {}), ("snap15", "snap", dict(grow=.15)), ("snap60", "snap", dict(grow=.6)), ("sparkcrop", "sparkcrop", {})):
            sb = sd.seed_boxes(st, cl, method, fr, **kw)
            tr = track(f"{st}__{name}_960", st, [(o, f, b) for o, f, b, _ in sb], res=960)
            for lab, chk in (("raw", None), ("tread", make_check(st, tr, fr))):
                rows, false = score(st, tr, chk)
                print(st, name, lab, summary(rows, ("main",)), "false", len(false), [(r["stair"], r["floor"], r["present"], [c[3] for c in r["claims"]]) for r in rows if not r["ok"]], flush=True)
            if st != "mml_capitol" and method != "loose":
                for (o, f, b, fb), c in zip(sb, cl):
                    own = assign(st, c["seed_box"])
                    if not own or fb:
                        continue
                    x1, y1, x2, y2 = c["seed_box"]; R_ = (max(0, int(x1 - 30)), max(0, int(y1 - 30)), int(x2 + 30), int(y2 + 30))
                    tiles.append((name, own[0], vis.grid_crop(fr[f], R_, 1.6, 100, [(c["seed_box"], (0, 200, 255), name + " " + own[0]), (b, (0, 170, 0), "")])))
    W = 0; rows_ = []; line = []
    for _, _, t in tiles:
        t = cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if W + t.shape[1] > 1900:
            rows_.append(line); line = []; W = 0
        line.append(t); W += t.shape[1]
    rows_.append(line)
    out = []
    for l in rows_:
        h = max(t.shape[0] for t in l)
        v = np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in l])
        out.append(cv2.copyMakeBorder(v, 0, 0, 0, max(0, 1900 - v.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))[:, :1900])
    cv2.imwrite(str(vis.V / "dev_seeds.jpg"), np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("spent", round(m.spent("tightseed"), 4))
