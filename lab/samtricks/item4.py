"""Item 4: revision clouds and delta marks on the BCOE addendum sheets. Usage: item4.py whole | sam | tiles | tiles2 | fix | score"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/samtricks")
import api as a  # noqa: E402
import cv2  # noqa: E402
import fitz  # noqa: E402
import numpy as np  # noqa: E402
import reg  # noqa: E402
import rules as R  # noqa: E402
import vector_clouds as vc  # noqa: E402

fitz.TOOLS.mupdf_display_errors(False)
T = json.load(open(reg.TRUTH / "item4_clouds.json"))
PAGES = {"bcoe1": 1, "bcoe2": 2, "bcoe3": 3}
W, H = 3024.0, 2160.0
OUT = reg.OUT / "item4.json"
res = json.load(open(OUT)) if OUT.exists() else {}
TD = reg.OUT / "tiles"
TD.mkdir(exist_ok=True)

BOX = {k: {"type": "integer"} for k in ("x1", "y1", "x2", "y2")}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["clouds", "deltas"], "properties": {
    "clouds": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["x1", "y1", "x2", "y2", "delta"], "properties": {**BOX, "delta": {"type": "string"}}}},
    "deltas": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["x1", "y1", "x2", "y2", "number"], "properties": {**BOX, "number": {"type": "string"}}}}}}
PROMPT = ("This image is from a sheet of a building drawing set, part of a revised submittal. A revision cloud is a closed outline made of small scalloped arcs, drawn around a changed area. "
          "A delta is a small triangle holding a revision number, placed next to a cloud or in a revision list. List every revision cloud and every delta triangle in this image, however small. "
          "For each cloud give its box and the number in the nearest delta, or an empty string if there is none. For each delta give its box and the number inside it. "
          "Coordinates are integers on a 0-1000 grid of this image, x from the left edge and y from the top edge. Return empty lists if there are none.")


def tiles(cols=4, rows=3, overlap=.1):
    tw, th = W / cols * (1 + overlap), H / rows * (1 + overlap)
    return [(float(x), float(y), float(x + tw), float(y + th)) for x in np.linspace(0, W - tw, cols) for y in np.linspace(0, H - th, rows)]


def render(key, clip):
    # The name holds the far corner too: the first version keyed on x0, y0 only, so tile 0 collided with the whole sheet.
    p = TD / f"{key}_{int(clip[0])}_{int(clip[1])}_{int(clip[2])}_{int(clip[3])}.png"
    if not p.exists():
        pg = fitz.open(vc.PDF)[PAGES[key]]
        z = 2000 / max(clip[2] - clip[0], clip[3] - clip[1])
        pm = pg.get_pixmap(matrix=fitz.Matrix(z, z), clip=fitz.Rect(*clip), alpha=False)
        cv2.imwrite(str(p), cv2.cvtColor(np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n), cv2.COLOR_RGB2BGR))
    return cv2.imread(str(p))


def grid_pts(b, clip):
    """A 0-1000 box on a clip, as sheet points."""
    cw, ch = clip[2] - clip[0], clip[3] - clip[1]
    return [clip[0] + b[0] / 1000 * cw, clip[1] + b[1] / 1000 * ch, clip[0] + b[2] / 1000 * cw, clip[1] + b[3] / 1000 * ch]


def spark_on(key, clip, effort):
    im = render(key, clip)
    r, info = a.chat([a.text(PROMPT), a.image(im)], schema=SCHEMA, effort=effort, kind="item4")
    return dict(clouds=[dict(box=grid_pts([c["x1"], c["y1"], c["x2"], c["y2"]], clip), delta=c["delta"]) for c in r["clouds"]],
                deltas=[dict(box=grid_pts([c["x1"], c["y1"], c["x2"], c["y2"]], clip), number=c["number"]) for c in r["deltas"]], info=info)


def run_whole():
    def one(k):
        return k, spark_on(k, (0, 0, W, H), "medium")
    res["spark_whole"] = dict(a.pmap(one, list(PAGES), 3))
    OUT.write_text(json.dumps(res))


def run_sam():
    jobs = [(k, w, c) for k in PAGES for w in ("revision cloud", "cloud", "triangle") for c in tiles()]

    def one(j):
        k, w, c = j
        im = render(k, c)
        s = (c[2] - c[0]) / im.shape[1]
        return k, w, [dict(box=[c[0] + r["box"][0] * s, c[1] + r["box"][1] * s, c[0] + r["box"][2] * s, c[1] + r["box"][3] * s], conf=r["conf"]) for r in a.sam(w, im)]
    out = {}
    for k, w, rows in a.pmap(one, jobs, 8):
        out.setdefault(k, {}).setdefault(w, []).extend(rows)
    res["sam_tiles"] = out
    OUT.write_text(json.dumps(res))


def run_tiles(effort, name="spark_tiles"):
    jobs = [(k, c) for k in PAGES for c in tiles()]

    def one(j):
        k, c = j
        return k, spark_on(k, c, effort)
    out = {}
    for k, r in a.pmap(one, jobs, 6):
        o = out.setdefault(k, dict(clouds=[], deltas=[], usd=0))
        o["clouds"] += r["clouds"]
        o["deltas"] += r["deltas"]
        o["usd"] += r["info"]["usd"]
    res[name] = out
    OUT.write_text(json.dumps(res))


def fix_tile0():
    """Tile 0 of every sheet was sent as the whole sheet. Its boxes sit first in each list, so drop that leading group and add a real tile 0 call."""
    c = tiles()[0]
    res["bugged"] = json.loads(json.dumps({n: res[n] for n in ("spark_tiles", "spark_tiles2")}))
    for name in ("spark_tiles", "spark_tiles2"):
        for k in PAGES:
            o = res[name][k]
            for f in ("clouds", "deltas"):
                n = 0
                while n < len(o[f]) and o[f][n]["box"][2] <= c[2] + 1 and o[f][n]["box"][3] <= c[3] + 1:
                    n += 1
                print(name, k, f, "dropped", n, "of", len(o[f]), [[round(v) for v in x["box"]] for x in o[f][:n]])
                o[f] = o[f][n:]
            r = spark_on(k, c, "low")
            o["clouds"] += r["clouds"]
            o["deltas"] += r["deltas"]
            o["usd"] += r["info"]["usd"]
    OUT.write_text(json.dumps(res))


def nms(boxes):
    """Drop boxes repeated across tile overlaps."""
    keep = []
    for b in sorted(boxes, key=lambda q: -(q[2] - q[0]) * (q[3] - q[1])):
        if all(a.iou(b, k) < .3 for k in keep):
            keep.append(b)
    return keep


def red_boxes(key):
    pg = fitz.open(vc.PDF)[PAGES[key]]
    pm = pg.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    im = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)
    red = ((im[..., 0] > 150) & (im[..., 1] < 90) & (im[..., 2] < 90)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(red, np.ones((5, 5), np.uint8)))
    return [[float(st[i][0]), float(st[i][1]), float(st[i][0] + st[i][2]), float(st[i][1] + st[i][3])] for i in range(1, n) if max(st[i][2], st[i][3]) >= 8]


def score_clouds(key, preds):
    truth = T["clouds"][key]
    hits = a.match(preds, [c["box"] for c in truth], R.IOU_C)
    hp = {i for i, _, _ in hits}
    ht = {j for _, j, _ in hits}
    false = [p for i, p in enumerate(preds) if i not in hp and not any(a.iou(p, c["box"]) >= .1 for c in truth)]
    return dict(n=len(truth), hit=len(hits), false=len(false), preds=len(preds), missed=[truth[j]["id"] for j in range(len(truth)) if j not in ht])


def score_deltas(key, preds):
    truth = T["deltas"][key]
    cs = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    pairs = sorted((np.hypot(cs(p)[0] - cs(t["box"])[0], cs(p)[1] - cs(t["box"])[1]), i, j) for i, p in enumerate(preds) for j, t in enumerate(truth))
    up, ut = set(), set()
    for d, i, j in pairs:
        if d <= R.DELTA_R and i not in up and j not in ut:
            up.add(i)
            ut.add(j)
    return dict(n=len(truth), hit=len(ut), false=len(preds) - len(up), preds=len(preds))


def table(name, fn_c, fn_d=None):
    tc = dict(n=0, hit=0, false=0, preds=0)
    td = dict(n=0, hit=0, false=0, preds=0)
    miss = []
    per = {}
    for k in PAGES:
        s = score_clouds(k, fn_c(k))
        per[k] = f"{s['hit']}/{s['n']} false {s['false']}"
        miss += s["missed"]
        for f in tc:
            tc[f] += s[f]
        if fn_d:
            s = score_deltas(k, fn_d(k))
            for f in td:
                td[f] += s[f]
    print(f"{name:26s} clouds {tc['hit']}/{tc['n']} false {tc['false']} of {tc['preds']} {per} missed {miss}" + (f"\n{'':26s} deltas {td['hit']}/{td['n']} false {td['false']} of {td['preds']}" if fn_d else ""))
    return dict(clouds=tc, deltas=td if fn_d else None, per=per, missed=miss)


def overlay(name, fn_c, fn_d):
    cells = []
    for k in PAGES:
        im = render(k, (0, 0, W, H)).copy()
        s = im.shape[1] / W
        P = lambda b: ((int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)))
        for c in T["clouds"][k]:
            cv2.rectangle(im, *P(c["box"]), (255, 0, 255), 3)
        for c in T["deltas"][k]:
            cv2.rectangle(im, *P(c["box"]), (255, 160, 0), 2)
        for b in fn_c(k):
            cv2.rectangle(im, *P(b), (0, 170, 0), 2)
        for b in fn_d(k):
            cv2.rectangle(im, *P(b), (0, 120, 255), 2)
        cells.append(im)
    cv2.imwrite(str(reg.OUT / name), np.vstack(cells), [cv2.IMWRITE_JPEG_QUALITY, 80])


def ink_share(key, b):
    pg = fitz.open(vc.PDF)[PAGES[key]]
    pm = pg.get_pixmap(matrix=fitz.Matrix(1, 1), clip=fitz.Rect(*b), alpha=False)
    g = cv2.cvtColor(np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n), cv2.COLOR_RGB2GRAY)
    return float((g < 200).mean()) if g.size else 0.0


def agree(xs, ys):
    return [x for x in xs if any(a.iou(x, y) >= .1 for y in ys)]


def post_hoc():
    """Written after the first tile run was seen. Gates fixed here, not tuned on truth: ink share >= 1 percent, and a box must recur in the second run at IoU >= 0.1."""
    s1, s2 = res["spark_tiles"], res["spark_tiles2"]
    b1 = lambda k: nms([c["box"] for c in s1[k]["clouds"]])
    b2 = lambda k: nms([c["box"] for c in s2[k]["clouds"]])
    d1 = lambda k: nms([c["box"] for c in s1[k]["deltas"]])
    d2 = lambda k: nms([c["box"] for c in s2[k]["deltas"]])
    out = {}
    print("post hoc, beside the frozen score above")
    out["run2"] = table("Spark tiles, second run", b2, d2)
    print("  usd", {k: round(v["usd"], 4) for k, v in s2.items()})
    ink = lambda f: (lambda k: [b for b in f(k) if ink_share(k, b) >= .01])
    out["run1_ink"] = table("run 1 + ink gate", ink(b1), ink(d1))
    out["run2_ink"] = table("run 2 + ink gate", ink(b2), ink(d2))
    out["agree"] = table("both runs agree", lambda k: agree(b1(k), b2(k)), lambda k: agree(d1(k), d2(k)))
    out["agree_ink"] = table("both runs agree + ink gate", ink(lambda k: agree(b1(k), b2(k))), ink(lambda k: agree(d1(k), d2(k))))
    out["union_ink"] = table("either run + ink gate", ink(lambda k: nms(b1(k) + b2(k))), ink(lambda k: nms(d1(k) + d2(k))))
    # number read on matched deltas, run 1
    ok = n = 0
    for k in PAGES:
        tr = T["deltas"][k]
        for c in s1[k]["deltas"]:
            cx, cy = (c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2
            best = min(tr, key=lambda t: np.hypot((t["box"][0] + t["box"][2]) / 2 - cx, (t["box"][1] + t["box"][3]) / 2 - cy))
            if np.hypot((best["box"][0] + best["box"][2]) / 2 - cx, (best["box"][1] + best["box"][3]) / 2 - cy) <= R.DELTA_R:
                n += 1
                ok += c["number"].strip() in best["label"]
    print("delta numbers read right, run 1:", ok, "of", n, "matched boxes, tile overlaps counted twice")
    return out


def score():
    out = {}
    print("wrong pass here is a cloud the method does not report")
    out["red"] = table("red pixels (no model)", red_boxes, red_boxes)
    out["vector"] = table("vector arc chains", lambda k: [c["box"] for c in vc.candidates(PAGES[k])])
    if "spark_whole" in res:
        sw = res["spark_whole"]
        out["spark_whole"] = table("Spark whole sheet, medium", lambda k: [c["box"] for c in sw[k]["clouds"]], lambda k: [c["box"] for c in sw[k]["deltas"]])
        print("  usd", {k: round(v["info"]["usd"], 4) for k, v in sw.items()})
    if "sam_tiles" in res:
        st = res["sam_tiles"]
        for w in ("revision cloud", "cloud"):
            out[f"sam_{w}"] = table(f"SAM tiles '{w}'", lambda k: nms([r["box"] for r in st[k][w]]), lambda k: nms([r["box"] for r in st[k]["triangle"]]))
    if "spark_tiles" in res:
        s = res["spark_tiles"]
        out["spark_tiles"] = table("Spark 12 tiles a sheet", lambda k: nms([c["box"] for c in s[k]["clouds"]]), lambda k: nms([c["box"] for c in s[k]["deltas"]]))
        print("  usd", {k: round(v["usd"], 4) for k, v in s.items()})
        overlay("item4_spark_tiles.jpg", lambda k: nms([c["box"] for c in s[k]["clouds"]]), lambda k: nms([c["box"] for c in s[k]["deltas"]]))
    if "sam_tiles" in res:
        overlay("item4_sam_tiles.jpg", lambda k: nms([r["box"] for r in res["sam_tiles"][k]["cloud"]]), lambda k: [])
    if "spark_tiles2" in res:
        out["post_hoc"] = post_hoc()
    (reg.OUT / "item4_score.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "whole":
        run_whole()
    if c == "sam":
        run_sam()
    if c == "tiles":
        run_tiles(sys.argv[2] if len(sys.argv) > 2 else "low")
    if c == "tiles2":
        run_tiles("low", "spark_tiles2")
    if c == "fix":
        fix_tile0()
        run_sam()
    score()
    print("spent", round(a.m.spent("samtricks"), 3))
