"""Item 1: SAM concept "floor plan" against Spark's plan box and two no-model baselines. Usage: item1.py dev | held | score"""
import json
import sys

sys.path.insert(0, "lab/samtricks")
import api as a  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import reg  # noqa: E402
import rules as R  # noqa: E402

T = json.load(open(reg.TRUTH / "item1_viewports.json"))["sheets"]
OUT = reg.OUT / "item1.json"
res = json.load(open(OUT)) if OUT.exists() else {}
# Copied from lab/align/sheet_read.py, which runs on import.
box = {"type": "object", "additionalProperties": False, "required": ["x1", "y1", "x2", "y2"], "properties": {k: {"type": "integer"} for k in ["x1", "y1", "x2", "y2"]}}
ONE = {"type": "object", "additionalProperties": False, "required": ["printed_title", "printed_floor_label", "storey_rank", "plan_box", "sheet_rotation"],
       "properties": {"printed_title": {"type": "string"}, "printed_floor_label": {"type": "string"}, "storey_rank": {"type": "integer"}, "plan_box": box,
                      "sheet_rotation": {"type": "integer", "enum": [0, 90, 180, 270]}}}
P_ONE = ("This is one sheet of an architectural drawing set. It may be in any language. Report: "
         "printed_title, the sheet or building title as printed. printed_floor_label, the floor name exactly as printed. "
         "storey_rank, an integer for that floor: basement or cellar -1, sub-basement -2, the lowest full storey above ground 0, the next one up 1, and so on. Attic or roof uses the next rank above the top storey. "
         "plan_box, the box around the walls of the main floor plan only: exclude the title block, sheet border, legend, notes, schedules, dimension strings, grid bubbles, key plans, detail insets and site work. "
         "sheet_rotation, how many degrees clockwise the sheet must turn so that the text reads upright. "
         "Coordinates are integers on a 0-1000 grid of this image, x from the left edge and y from the top edge.")


def run(k):
    if k in res:
        return k, res[k]
    im = reg.sheet(k)
    h, w = im.shape[:2]
    s = a.sam("floor plan", im)
    r, info = a.m.spark(P_ONE, [im], schema=ONE)
    b = r["plan_box"]
    return k, dict(sam=[dict(box=a.to_grid(x["box"], w, h), conf=x["conf"]) for x in s], spark=[b["x1"], b["y1"], b["x2"], b["y2"]], read={x: r[x] for x in ("printed_title", "printed_floor_label", "storey_rank", "sheet_rotation")}, usd=info["usd"], ink=a.largest_ink_box(im))


def score(k, preds):
    """preds is a list of grid boxes. Returns counts for one sheet."""
    t = T[k]
    req, opt = t["req"], t["opt"]
    hits = a.match(preds, req, R.IOU_V)
    hit_p = {i for i, _, _ in hits}
    free = [i for i in range(len(preds)) if i not in hit_p]
    false = [i for i in free if not any(a.iou(preds[i], o) >= R.IOU_V for o in opt)]
    clipped = [j for i, j, _ in hits if a.covered(req[j], preds[i]) < R.CLIP]
    return dict(n_req=len(req), hit=len(hits), miss=len(req) - len(hits), false=len(false), clipped=len(clipped), kind=t["kind"], cover=[max([a.covered(q, p) for p in preds], default=0) for q in req])


def sam_boxes(k, floor):
    return [x["box"] for x in res[k]["sam"] if (x["conf"] or 0) >= floor]


def agg(keys, fn):
    rows = [score(k, fn(k)) for k in keys]
    s = dict(req=sum(r["n_req"] for r in rows), hit=sum(r["hit"] for r in rows), false=sum(r["false"] for r in rows), clipped=sum(r["clipped"] for r in rows))
    s["false_on_noplan"] = sum(r["false"] for r in rows if r["kind"] == "none")
    s["noplan_sheets_false"] = sum(1 for r in rows if r["kind"] == "none" and r["false"])
    s["noplan_sheets"] = sum(1 for r in rows if r["kind"] == "none")
    s["clean_sheets"] = sum(1 for r in rows if r["miss"] == 0 and r["false"] == 0)
    s["sheets"] = len(rows)
    s["recall_single"] = f"{sum(r['hit'] for r in rows if r['kind'] == 'single')}/{sum(r['n_req'] for r in rows if r['kind'] == 'single')}"
    s["recall_several"] = f"{sum(r['hit'] for r in rows if r['kind'] == 'several')}/{sum(r['n_req'] for r in rows if r['kind'] == 'several')}"
    s["mean_cover"] = round(float(np.mean([c for r in rows for c in r["cover"]])), 3) if s["req"] else None
    return s


def f1(s):
    p = s["hit"] / max(s["hit"] + s["false"], 1)
    r = s["hit"] / max(s["req"], 1)
    return 2 * p * r / max(p + r, 1e-9)


def overlay(keys, floor, name):
    cells = []
    for k in keys:
        im = reg.sheet(k).copy()
        h, w = im.shape[:2]
        for q in T[k]["req"]:
            x = a.to_px(q, w, h)
            cv2.rectangle(im, x[:2], x[2:], (255, 0, 255), 3)
        for q in T[k]["opt"]:
            x = a.to_px(q, w, h)
            cv2.rectangle(im, x[:2], x[2:], (255, 0, 255), 1)
        x = a.to_px(res[k]["spark"], w, h)
        cv2.rectangle(im, x[:2], x[2:], (255, 80, 0), 3)
        for b in sam_boxes(k, floor):
            x = a.to_px(b, w, h)
            cv2.rectangle(im, x[:2], x[2:], (0, 170, 0), 4)
        c = cv2.resize(im, (640, 440), interpolation=cv2.INTER_AREA)
        cv2.putText(c, k, (6, 22), 0, .7, (0, 0, 255), 2)
        cells.append(c)
    while len(cells) % 4:
        cells.append(np.full_like(cells[0], 255))
    cv2.imwrite(str(reg.OUT / name), np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 80])


def table(keys, floor, label):
    print(f"== {label}, SAM conf floor {floor}")
    for name, fn in (("SAM", lambda k: sam_boxes(k, floor)), ("Spark", lambda k: [res[k]["spark"]]), ("left 78%", lambda k: [[0, 0, 780, 1000]]), ("largest ink", lambda k: [res[k]["ink"]])):
        print(f"{name:12s}", agg(keys, fn))
    ious = [a.iou(max(sam_boxes(k, floor), key=lambda b: a.iou(b, res[k]["spark"])), res[k]["spark"]) for k in keys if sam_boxes(k, floor)]
    print("SAM against Spark box, best IoU on sheets where SAM fired: median", round(float(np.median(ious)), 3), "n", len(ious), "IoU>=.5", sum(v >= .5 for v in ious))
    print("per sheet (SAM boxes, hits, false | Spark hit, false):")
    for k in keys:
        s1, s2 = score(k, sam_boxes(k, floor)), score(k, [res[k]["spark"]])
        print(f"  {k:6s} {T[k]['kind']:8s} req {s1['n_req']:2d} SAM n {len(sam_boxes(k, floor)):2d} hit {s1['hit']:2d} false {s1['false']} clipped {s1['clipped']} confs {[x['conf'] for x in res[k]['sam']][:6]} | Spark hit {s2['hit']} false {s2['false']} clipped {s2['clipped']}")


if __name__ == "__main__":
    which = sys.argv[1]
    if which in ("dev", "held"):
        keys = reg.DEV if which == "dev" else reg.HELD
        for k, v in a.pmap(run, keys):
            res[k] = v
        OUT.write_text(json.dumps(res, indent=0))
        print("spent", round(a.m.spent("samtricks"), 3))
    if which == "dev":
        best = max(R.CONF_GRID, key=lambda c: (round(f1(agg(reg.DEV, lambda k: sam_boxes(k, c))), 6), -c))
        for c in R.CONF_GRID:
            print(c, round(f1(agg(reg.DEV, lambda k: sam_boxes(k, c))), 3), agg(reg.DEV, lambda k: sam_boxes(k, c)))
        (reg.OUT / "item1_floor.json").write_text(json.dumps(dict(floor=best)))
        table(reg.DEV, best, "dev")
        overlay(reg.DEV, best, "item1_dev.jpg")
    if which == "score":
        floor = json.load(open(reg.OUT / "item1_floor.json"))["floor"]
        table(reg.HELD, floor, "held-out")
        overlay(reg.HELD, floor, "item1_held.jpg")
