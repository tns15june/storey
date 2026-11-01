"""Vector stair runs per floor, mapped into stack px, clustered like Spark boxes. No API. wfu_huffman and mml_abbot only."""
import sys, json
sys.path.insert(0, "lab/tightseed"); sys.path.insert(0, "lab/modern")
from core import *
import reg, vec, pymupdf as fitz


def corners(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


def wfu():
    H = {int(k): np.array(v) for k, v in json.load(open(ST / "wfu_huffman/H.json"))["H"].items()}
    A = json.load(open(ROOT / "data/lab/audit/wfu_huffman/align.json"))
    items = []
    for f in range(4):
        pg = vec.open_page("wfu_huffman", "HUFFMAN-FLOOR-PLAN-G-3.pdf", f)
        z = 2000 / pg.rect.width
        M = pg.rotation_matrix
        segs, _ = vec.segments(pg)
        Af = np.vstack([A[str(f)]["A"], [0, 0, 1]]) if "A" in A[str(f)] else np.eye(3)
        for r in vec.stair_runs(segs, tread_range=(4, 9)):
            rect = fitz.Rect(r["box"]) * M
            p = reg.apply(H[f] @ Af, [[x * z, y * z] for x, y in corners((rect.x0, rect.y0, rect.x1, rect.y1))])
            items.append((f, [float(v) for v in (*p.min(0), *p.max(0))]))
    return items


def abbot():
    H = {int(k): np.array(v) for k, v in json.load(open(ST / "mml_abbot/H.json"))["H"].items()}
    pg = vec.open_page("mml_abbot", "THE-ABBOT.pdf", 1)
    segs, _ = vec.segments(pg)
    z = 1.744
    pan = {0: (84, 872, 1231, 1695), 1: (1241, 872, 2386, 1695), 2: (1241, 40, 2386, 862), 3: (84, 40, 1231, 862)}
    ipp = vec.inches_per_pt('1/4" = 1\'-0"')
    runs = vec.stair_runs(segs, tread_range=(9 / ipp, 14 / ipp))
    runs = [r for r in runs if 18 <= r["width"] * ipp <= 130]
    items = []
    M = pg.rotation_matrix
    for r in runs:
        q = fitz.Rect(r["box"]) * M
        cx, cy = (q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2
        for f, (x1, y1, x2, y2) in pan.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                p = reg.apply(H[f], [[(x - x1) * z, (y - y1) * z] for x, y in corners((q.x0, q.y0, q.x1, q.y1))])
                items.append((f, [float(v) for v in (*p.min(0), *p.max(0))]))
    return items


if __name__ == "__main__":
    for name, fn in (("wfu_huffman", wfu), ("mml_abbot", abbot)):
        it = fn()
        print(name, len(it), [(f, [round(v) for v in b]) for f, b in it][:24])
        json.dump(it, open(R / f"vec_{name}.json", "w"))
