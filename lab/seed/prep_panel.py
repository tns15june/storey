"""For sheets where the plan is a small panel: Spark finds the plan panel, crop it from a 2.5x render, align crops. Replaces that building's frames."""
import sys; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab")
from common import *
import metalab as m
from seed_spark import SCHEMA, TAIL

ZOOM = 2.5
PROMPT = ('You are an object grounding expert. This is a sheet of architectural drawings. Provide the bounding box coordinates of the object: "the main floor plan of the whole building". '
          'Include exterior stairs and porches drawn with it. Exclude the title block, schedules, details, enlarged plans, key plans, notes and legends. Return exactly one object.' + TAIL)
OUT_W = 2000


def render(bid, f):
    p = PLANS / bid / f["file"]
    pg = fitz.open(p)[f["page"]]
    z = W * ZOOM / pg.rect.width
    px = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
    return cv2.cvtColor(np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, 3), cv2.COLOR_RGB2BGR)


for bid in sys.argv[1:]:
    d = OUT / bid
    (d / "stack.mp4").unlink(missing_ok=True)
    fl = CAT[bid]["floors"]
    crops, boxes = [], []
    for i, f in enumerate(fl):
        cache = d / f"panel{i}.json"
        big = render(bid, f)
        if cache.exists():
            o = json.load(open(cache))
        else:
            r, info = m.spark(PROMPT, images=[load_floor(bid, f)], schema=SCHEMA)
            o = r["objects"][0]
            json.dump(o, open(cache, "w"))
        h, w = big.shape[:2]
        b = to_px(o, w, h)
        mx, my = round((b[2] - b[0]) * 0.06), round((b[3] - b[1]) * 0.06)
        b = [max(0, b[0] - mx), max(0, b[1] - my), min(w, b[2] + mx), min(h, b[3] + my)]
        boxes.append(b)
        crops.append(big[b[1]:b[3], b[0]:b[2]])
        print(bid, i, "panel px", b, "of", (w, h))
    W2, H2 = max(c.shape[1] for c in crops), max(c.shape[0] for c in crops)
    s = min(1.0, OUT_W / W2)
    can = []
    for c in crops:  # same drawing scale on every floor, centred on a common canvas
        c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        t = np.full((round(H2 * s), round(W2 * s), 3), 255, np.uint8)
        y, x = (t.shape[0] - c.shape[0]) // 2, (t.shape[1] - c.shape[1]) // 2
        t[y:y + c.shape[0], x:x + c.shape[1]] = c
        can.append(t)
    ref = can[0]
    info = {}
    for i, c in enumerate(can):
        al, mt = (ref, {"method": "ref", "overlap": 1.0}) if i == 0 else align(c, ref)
        cv2.imwrite(str(d / f"f{i}.png"), al)
        info[i] = mt
        print(bid, i, mt["method"], mt["overlap"], al.shape, "stair scale px per sheet px", round(ZOOM * s, 2))
    json.dump(info, open(d / "align.json", "w"), indent=1)
