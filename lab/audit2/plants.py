"""Soft plants: the stair symbol is removed but thick walls stay, so the room is still drawn. No API. Run with -I.
Targets come from a fixed rule, never from outcomes: every cls main stair of the presence held-out buildings on at least 2 floors, at its lower-median present floor,
where the clean evidence has an established stair row within 40 px of the truth box on an aligned floor. A stair with no such row is listed as untestable, because arm C cannot flag what it does not track."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "lab/presence")
sys.path.insert(0, "lab/defects")
import pres
from pres import OUT, SH, T, box_dist, reg
from plant import erase

AUD = pres.ROOT / "data/lab/audit2"
P = AUD / "plants"
THICK = 5  # px on a 2000 px sheet: a stroke this wide survives, so walls stay and treads go


def soft(im, box, seed=1, k=THICK):
    """Erase thin strokes inside box, keep thick ones. Returns the new sheet and the share of box ink removed."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = x1 + 2, y1 + 2, x2 - 2, y2 - 2
    ink = np.uint8(reg.ink(im) > 0)
    th = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    th = cv2.dilate(th, np.ones((5, 5), np.uint8))
    region = np.zeros(ink.shape, bool)
    region[y1:y2, x1:x2] = True
    region &= th == 0
    return erase(im, x1, y1, x2, y2, region, seed), float(ink[y1:y2, x1:x2][th[y1:y2, x1:x2] == 0].sum()) / max(float(ink[y1:y2, x1:x2].sum()), 1)


def targets():
    out, untestable = [], []
    for b in pres.HELD:
        E = json.load(open(OUT / f"cells/{b}.json"))
        for o, v in T[b]["objects"].items():
            if v["kind"] not in ("stair", "stairs") or v["cls"] != "main" or len(v["boxes"]) < 2:
                continue
            fl = sorted(map(int, v["boxes"]))
            f = fl[(len(fl) - 1) // 2]
            bx = v["boxes"][str(f)]
            rows = [(box_dist(r["cells"][str(f)]["chk"], bx), r) for r in E["rows"] if r["fam"] == "stair" and r["established"] and r["cells"][str(f)]["chk"] and r["cells"][str(f)]["aligned"]]
            rows = sorted([x for x in rows if x[0] <= 40], key=lambda x: x[0])
            if not rows:
                untestable.append(dict(b=b, obj=o, f=f, why="no established row within 40 px on an aligned floor"))
                continue
            r = rows[0][1]
            out.append(dict(b=b, obj=o, f=f, box=bx, row=r["id"], chk=r["cells"][str(f)]["chk"], names=r["names"][:2], v_clean=r["cells"][str(f)]["v"]))
    return out, untestable


if __name__ == "__main__":
    P.mkdir(parents=True, exist_ok=True)
    tg, un = targets()
    tiles = []
    for t in tg:
        im = cv2.imread(str(SH / f"{t['b']}/{t['f']}.jpg"))
        new, share = soft(im, t["box"])
        t["share_removed"] = round(share, 3)
        d = P / f"{t['b']}__{t['obj']}"
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / f"{t['f']}.jpg"), new, [cv2.IMWRITE_JPEG_QUALITY, 92])
        x1, y1, x2, y2 = [int(v) for v in t["box"]]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = max(x2 - x1, y2 - y1) // 2 + 50
        cut = lambda a: cv2.resize(cv2.copyMakeBorder(a, r, r, r, r, cv2.BORDER_CONSTANT, value=(180, 180, 180))[cy:cy + 2 * r, cx:cx + 2 * r], (250, 250), interpolation=cv2.INTER_CUBIC)
        q = np.hstack([cut(im), cut(new)])
        cv2.putText(q, f"{t['b'][:14]} {t['obj']} f{t['f']} rm {share:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .38, (200, 0, 0), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(q, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    n = 4
    rows = [np.hstack(tiles[i:i + n] + [np.zeros_like(tiles[0])] * (n - len(tiles[i:i + n]))) for i in range(0, len(tiles), n)]
    cv2.imwrite(str(AUD / "plants_soft.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(dict(targets=tg, untestable=un), open(AUD / "plant_targets.json", "w"), indent=1)
    print(len(tg), "targets", len(un), "untestable", un)
