"""Dev only, no API: snap on the loose seeds, overlays, and tread scores on present and absent floors."""
import sys, json
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import dev1, tread, vis


def run(st, zoom=2):
    cl = dev1.loose_clusters(st)
    fr = frames(st); gr = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr]
    tiles = []; rows = []
    for i, c in enumerate(cl):
        own = assign(st, c["seed_box"])
        b, info = tread.snap(gr[c["seed_floor"]], c["seed_box"])
        t = TRUTH[st]["stairs"]
        tb = [t[k]["boxes"].get(str(c["seed_floor"])) for k in own]
        tb = [x for x in tb if x]
        l_iou = max([iou(c["seed_box"], x) for x in tb], default=None); s_iou = max([iou(b, x) for x in tb], default=None) if b else None
        rows.append((i + 1, own, c["seed_floor"], [round(v) for v in c["seed_box"]], [round(v) for v in b] if b else None, l_iou and round(l_iou, 2), s_iou and round(s_iou, 2), [(round(p, 1), round(a)) for p, a, _ in info.get("modes", [])]))
        x1, y1, x2, y2 = c["seed_box"]; g = 40
        R = (max(0, int(x1 - g)), max(0, int(y1 - g)), min(fr[0].shape[1], int(x2 + g)), min(fr[0].shape[0], int(y2 + g)))
        m_ = [(c["seed_box"], (0, 200, 255), f"{i+1} {','.join(own)}")]
        if b: m_.append((b, (0, 180, 0), ""))
        m_ += [(x, (255, 0, 255), "") for x in tb]
        tiles.append(vis.grid_crop(fr[c["seed_floor"]], R, zoom, 100, m_))
    for r in rows:
        print(r)
    # montage
    H = max(t.shape[0] for t in tiles); W = 0; line = []; lines = []
    for t in tiles:
        t = cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if W + t.shape[1] > 1800:
            lines.append(line); line = []; W = 0
        line.append(t); W += t.shape[1]
    lines.append(line)
    out = []
    for l in lines:
        h = max(t.shape[0] for t in l)
        l = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in l]
        v = np.hstack(l)
        out.append(cv2.copyMakeBorder(v, 0, 0, 0, 1800 - v.shape[1] if v.shape[1] < 1800 else 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))[:, :1800])
    p = vis.V / f"dev_snap_{st}.jpg"
    cv2.imwrite(str(p), np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(p)


if __name__ == "__main__":
    for st, z in (("bme_f", 2), ("habs_stelizabeths_center", 2.2), ("mml_capitol", .7)):
        print("==", st)
        run(st, z)
