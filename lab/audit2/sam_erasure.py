"""L2: does SAM keep a mask on a floor where a stair was erased mid-run? The tightseed recipe lets a SAM mask cancel an ending flag, and every absence it tested sat at the top of a run.
Usage: sam_erasure.py build | run | score. build makes frames, no API. Seeds are hand boxes pinned on every present floor except the erased one, the best case for the recipe.
Kinds: clean (control), hard (lab/defects erase of the whole box, the presence plant), soft (lab/audit2/plants.soft, walls stay)."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/audit2")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/defects")
import core
import plants
from plant import erase

AUD = core.ROOT / "data/lab/audit2"
core.SAMD = AUD / "sam"
core.SAMD.mkdir(exist_ok=True)
STACKS = ("bme_k", "habs_va_bldg1", "wfu_huffman", "queens_royce1", "leonard4")
M = 2  # erased floor


def stairs(st):
    return [k for k, s in core.TRUTH[st]["stairs"].items() if s["cls"] == "main" and {M - 1, M, M + 1} <= {int(f) for f in s["boxes"]}]


def seeds(st):
    return [(i + 1, f, core.TRUTH[st]["stairs"][k]["boxes"][str(f)]) for i, k in enumerate(stairs(st)) for f in sorted(int(x) for x in core.TRUTH[st]["stairs"][k]["boxes"]) if f != M]


def frames(st, kind):
    frs = core.frames(st)
    if kind == "clean":
        return frs
    im = frs[M]
    for i, k in enumerate(stairs(st)):
        b = core.TRUTH[st]["stairs"][k]["boxes"][str(M)]
        x1, y1, x2, y2 = [int(v) for v in b]
        im = erase(im, x1 + 2, y1 + 2, x2 - 2, y2 - 2, seed=i + 1) if kind == "hard" else plants.soft(im, b, seed=i + 1)[0]
    frs[M] = im
    return frs


def build():
    tiles = []
    for st in STACKS:
        print(st, stairs(st))
        a, b = frames(st, "hard")[M], frames(st, "soft")[M]
        for k in stairs(st)[:3]:
            x1, y1, x2, y2 = [int(v) for v in core.TRUTH[st]["stairs"][k]["boxes"][str(M)]]
            r = max(x2 - x1, y2 - y1) // 2 + 40
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cut = lambda q: cv2.resize(cv2.copyMakeBorder(q, r, r, r, r, cv2.BORDER_CONSTANT, value=(180, 180, 180))[cy:cy + 2 * r, cx:cx + 2 * r], (220, 220), interpolation=cv2.INTER_CUBIC)
            t = np.hstack([cut(core.frames(st)[M]), cut(a), cut(b)])
            cv2.putText(t, f"{st[:10]} {k} clean hard soft", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .38, (200, 0, 0), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    n = 2
    cv2.imwrite(str(AUD / "sam_plants.jpg"), np.vstack([np.hstack(tiles[i:i + n] + [np.zeros_like(tiles[0])] * (n - len(tiles[i:i + n]))) for i in range(0, len(tiles), n)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def run():
    for st in STACKS:
        for kind in ("clean", "hard", "soft"):
            core.track(f"{st}__erase_{kind}", st, seeds(st), frs=frames(st, kind))


def score():
    tot = {}
    for st in STACKS:
        for kind in ("clean", "hard", "soft"):
            tr = json.load(open(core.SAMD / f"{st}__erase_{kind}.json"))
            for i, k in enumerate(stairs(st)):
                mk = tr["tracks"].get(str(i + 1), {}).get(str(M))
                b = core.TRUTH[st]["stairs"][k]["boxes"][str(M)]
                on = core.has(mk)
                h = bool(on and core.hit(mk["box"], b))
                a = tot.setdefault(kind, dict(n=0, mask=0, on_box=0, rows=[]))
                a["n"] += 1
                a["mask"] += on
                a["on_box"] += h
                a["rows"].append((st, k, None if not mk else mk["conf"], h))
    for kind, a in tot.items():
        print(kind, "stairs", a["n"], "mask conf>=.5 on floor", M, a["mask"], "on the stair box", a["on_box"])
        print("   ", [(s[:6], k, c, h) for s, k, c, h in a["rows"]])
    json.dump(tot, open(AUD / "sam_erasure.json", "w"), indent=1)


if __name__ == "__main__":
    {"build": build, "run": run, "score": score}[sys.argv[1]]()
