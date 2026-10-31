"""Seed recipes. Each returns one box per cluster on the cluster's seed floor, in stack px, with the loose box as fallback when the method finds nothing."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import tread

CROP_P = ('You are an object grounding expert. This is a zoomed crop of an architectural floor plan. Find the run of parallel tread lines of the staircase nearest the centre of the crop: '
          'the evenly spaced parallel lines that mark the steps. Give one box tight around all the tread lines of that one staircase. Do not include walls, landings, door swings, the rest of the room, or labels. '
          'If the crop has no run of parallel tread lines, return no object.' + TAIL)


def crop_window(b, shape, k=2.2, minside=220):
    w, h = b[2] - b[0], b[3] - b[1]
    side_w, side_h = max(k * w, minside), max(k * h, minside)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x1, y1 = max(0, int(cx - side_w / 2)), max(0, int(cy - side_h / 2))
    x2, y2 = min(shape[1], int(cx + side_w / 2)), min(shape[0], int(cy + side_h / 2))
    return x1, y1, x2, y2


def spark_crop(a):
    """a = (stack, cluster index, floor, loose box, frame). One Spark call on a zoomed crop. Cached."""
    stack, i, f, b, fr = a
    p = SPK / f"{stack}_crop{i}_f{f}.json"
    if p.exists():
        return json.load(open(p))
    x1, y1, x2, y2 = crop_window(b, fr.shape)
    c = fr[y1:y2, x1:x2]
    z = min(4.0, 1000 / max(c.shape[:2]))
    c = cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA)
    q = R / "crops" / f"{stack}_crop{i}_f{f}.png"
    cv2.imwrite(str(q), c)
    r, info = m.spark(CROP_P, images=[str(q)], schema=SCHEMA)
    h, w = y2 - y1, x2 - x1
    bs = [[v + d for v, d in zip(to_px(o, w, h), (x1, y1, x1, y1))] for o in r["objects"]]
    o = dict(boxes=[[round(v, 1) for v in x] for x in bs], window=[x1, y1, x2, y2], info=info)
    json.dump(o, open(p, "w"))
    return o


def union(bs):
    a = np.array(bs)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max())]


def seed_boxes(stack, cl, method, fr=None, grow=.6):
    """Returns list of (obj id, floor, box, used_fallback)."""
    fr = fr or frames(stack)
    out = []
    if method == "loose":
        return [(i + 1, c["seed_floor"], c["seed_box"], False) for i, c in enumerate(cl)]
    if method == "snap":
        for i, c in enumerate(cl):
            g = cv2.cvtColor(fr[c["seed_floor"]], cv2.COLOR_BGR2GRAY)
            b, info = tread.snap(g, c["seed_box"], grow=grow)
            out.append((i + 1, c["seed_floor"], b or c["seed_box"], b is None))
        return out
    if method == "sparkcrop":
        jobs = [(stack, i, c["seed_floor"], c["seed_box"], fr[c["seed_floor"]]) for i, c in enumerate(cl)]
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(spark_crop, jobs))
        for i, (c, r) in enumerate(zip(cl, res)):
            out.append((i + 1, c["seed_floor"], union(r["boxes"]) if r["boxes"] else c["seed_box"], not r["boxes"]))
        return out
    raise ValueError(method)
