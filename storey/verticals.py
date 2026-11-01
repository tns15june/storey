"""Which stair is drawn on which floor.

The rule, prompts and constants are arm C of lab/presence, frozen before held-out, where 0 of 15 planted erasures stayed Found. Changing one needs a new measurement.
Every pixel constant is in pixels of a sheet with long edge 2000 px, so the evidence is read from 2000 px copies and results convert to sheet pixels on the way out.
SAM decides nothing: its masks are drawn on Found cells afterwards, as a picture of what was found."""
import functools
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from openai import APIError, BadRequestError

from .floors import WORK_PX, _resized, ink, warp_point
from .meta import BudgetReached, sam_video, spark

log = logging.getLogger(__name__)

RUNS = 3
INK_LOW = 0.40    # dev: real stairs 0.48 and up, erased stairs 0.27 and down, n 6 erased
INK_HIGH = 0.90   # dev: a Missing candidate with this much ink is worth a look, and stairs absent on dev rarely reach it
R = 80            # cluster radius: stairs sit 90 px or more apart, and Spark's points scatter less than that on the dev set
INK_HALF = 30
BLANK = 0.002     # ink density in a box of half side BLANK_HALF, under which nothing is drawn at a place
BLANK_HALF = 50
SIDE, UP, SNAP = 260, 3, 30   # a crop's side, its enlargement, and the grid its centre snaps to
WORKERS = 6
SEED_ROUNDS = ((3.0, 300), (1.8, 200))   # each window as a multiple of the last box, and its smallest side
SAM_OBJECTS = 16   # Meta's limit on prompted objects in one request
MASK_CONF = 0.5    # the confidence the lab scored masks at, which says how sure SAM is of its outline and nothing about whether a stair is there
MASK_REACH = 120   # a tracked mask further than this from the registered place has left the stair
SHAFTS = "Elevators and shafts are not followed."

# shortcut: a crop lost to an API failure stays Check, because a finished stage is not rerun without fresh, raise as _compare does if a real review loses crops that way.
BAD_REPLY = (ValueError, KeyError, TypeError, APIError)   # one failed call costs one run or one crop. BudgetReached passes through, so the pipeline records the stage and a rerun reads finished calls from the cache

FLOOR_P = """These images are floor plans of one building, one image per floor, in this order: {order}. Staircases in the same place on consecutive floors are the same staircase.
List every distinct staircase in the building. For each give a short name, and for every floor in the list say whether it is drawn on that floor (present) and its centre x, y in the 0-1000 range of that floor's image (0, 0 if absent). A stair drawn struck through or filled in counts as absent. Compare the floors carefully, stair by stair."""
EXTRA = """
Also list every elevator and every vertical shaft in a separate list called shafts, in the same form, with kind set to elevator or shaft. A shaft is a box drawn with a diagonal cross, a duct, a chimney or a ventilation shaft. Set floor to the floor's code, F1, F2 and so on."""
STR = {"type": "string"}
FL = {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["floor", "present", "x", "y"],
                                 "properties": {"floor": STR, "present": {"type": "boolean"}, "x": {"type": "integer"}, "y": {"type": "integer"}}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["stairs", "shafts"], "properties": {
    "stairs": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "floors"], "properties": {"name": STR, "floors": FL}}},
    "shafts": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "kind", "floors"],
                                          "properties": {"name": STR, "kind": {"type": "string", "enum": ["elevator", "shaft"]}, "floors": FL}}}}}
ANS = {"type": "object", "additionalProperties": False, "required": ["answer", "seen"],
       "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unclear"]}, "seen": {"type": "string"}}}
QUESTION = ("This is a crop from one floor plan of a building. Is a staircase drawn at or near the centre of this crop? A staircase is a run of closely spaced parallel lines (treads), straight or curved, "
            "often with an arrow or a break line. Walls, doors, hatching, dimension lines and text alone do not count. Answer yes, no or unclear, and say in a few words what is at the centre.")
TAIL = ' For each object, answer in the format {"object_name": "<name>", "x_min": <int>, "y_min": <int>, "x_max": <int>, "y_max": <int>}. The coordinates should be in the 0-1000 range, x from the left edge and y from the top edge of the image. If you cannot find an object, omit it.'
WHOLE_P = ('You are an object grounding expert. This is a zoomed crop of an architectural floor plan. Find the staircase nearest the centre of the crop and give one box tight around the whole stair symbol: '
           'all of its flights of tread lines and the landings between them, out to the stair walls. Do not include corridors, the rooms beside it, door swings or labels. If there is no staircase, return no object.' + TAIL)
BOX = {"type": "object", "additionalProperties": False, "required": ["object_name", "x_min", "y_min", "x_max", "y_max"], "properties": {"object_name": {"type": "string"}, **{k: {"type": "integer"} for k in ("x_min", "y_min", "x_max", "y_max")}}}
GROUND = {"type": "object", "additionalProperties": False, "required": ["objects"], "properties": {"objects": {"type": "array", "items": BOX}}}


def decide(r, c):
    """The pure rule. r: row with established and ink. c: its cell with v, crop, blank and aligned. Returns (state, reason).
    Checks only ever move a cell toward check, and never lift one to found. Found means a stair is drawn at this place on this floor, no more."""
    v, crop, blank, ratio = c["v"], c.get("crop"), c.get("blank"), r["ink"]
    if not r["established"]:
        if v == 0 or blank or crop == "no":
            return "not_followed", "single run, not confirmed"
        return "check", "seen by one run only"
    if not c["aligned"]:
        return "check", "floor unaligned"
    if v >= 2:
        if ratio is not None and ratio < INK_LOW:
            return "check", "ink gone at the registered place"
        if blank:
            return "check", "runs claim it, nothing is drawn"
        if crop is None:   # every such cell had an answer in the lab, and its rule would have read none as Found
            return "check", "crop gave no usable reply"
        if crop in ("no", "unclear"):
            return "check", f"crop says {crop}"
        return "found", f"{v} of 3 runs"
    if blank:
        return "missing", "nothing drawn here"
    if crop is None:   # a cell that is not blank always had a crop asked here, so this is a failed reply, and every such cell had an answer in the lab
        return "check", "crop gave no usable reply"
    if crop in ("yes", "unclear"):
        return "check", f"crop says {crop}"
    if ratio is not None and ratio >= INK_HIGH:
        return "check", "ink as dense as on floors with the stair"
    return "missing", f"{v} of 3 runs"


def floor_index(name, k, n, order):
    m_ = re.search(r"(\d+)", str(name))
    if m_ and 1 <= int(m_.group(1)) <= n and str(name).strip().upper().startswith("F"):
        return order[int(m_.group(1)) - 1]
    return order[k] if k < n else None  # shortcut: falls back to array position, only reached if a run ignores the codes


def cluster(points):
    """Leader clustering on reference position with two recentring passes. A row is a place, not a name, so a run that glues two places under one name is split.
    Points without a reference position join through their entity, and an entity seen only on unaligned floors becomes its own row."""
    placed = [p for p in points if p["ref"] is not None]
    cs = []
    for _ in range(3):
        for c in cs:
            c["pts"] = []
        for p in placed:
            best, bd = None, 1e9
            for c in cs:
                d = float(np.hypot(*(p["ref"] - c["pos"])))
                if d < bd:
                    best, bd = c, d
            if best is not None and bd <= R:
                best["pts"].append(p)
            else:
                cs.append(dict(pos=p["ref"].copy(), pts=[p]))
        cs = [c for c in cs if c["pts"]]
        for c in cs:
            c["pos"] = np.median([p["ref"] for p in c["pts"]], axis=0)
    home = {}
    for i, c in enumerate(cs):
        for p in c["pts"]:
            home.setdefault(p["ent"], []).append(i)
    # a stray point rejoins its own entity's row when that row is close: same name from Spark, and near enough to be the same stair
    maj = {e: max(set(h), key=h.count) for e, h in home.items()}
    for i, c in enumerate(cs):
        for p in list(c["pts"]):
            j = maj[p["ent"]]
            if j != i and float(np.hypot(*(p["ref"] - cs[j]["pos"]))) <= 2 * R:
                c["pts"].remove(p)
                cs[j]["pts"].append(p)
    cs = [c for c in cs if c["pts"]]
    home = {}
    for i, c in enumerate(cs):
        for p in c["pts"]:
            home.setdefault(p["ent"], []).append(i)
    orphans = {}
    for p in points:
        if p["ref"] is None:
            h = home.get(p["ent"])
            if h:
                cs[max(set(h), key=h.count)]["pts"].append(p)
            else:
                orphans.setdefault(p["ent"], []).append(p)
    for v in orphans.values():
        cs.append(dict(pos=None, pts=v))
    return cs


@functools.lru_cache(maxsize=50)   # one map per floor, and every row reads every floor twice
def _ink_map(path):
    return ink(cv2.imread(str(path))) > 0


def _density(path, chk, half):
    im = _ink_map(path)
    h, w = im.shape
    x, y = int(chk[0]), int(chk[1])
    c = im[max(y - half, 0):max(min(y + half, h), 0), max(x - half, 0):max(min(x + half, w), 0)]   # a place off the sheet holds nothing, and a negative slice end would read most of it
    return float(c.mean()) if c.size else 0.0


def ink_ratio(row, paths, half=INK_HALF):
    """paths: floor to its 2000 px copy. Each floor's ink in a box at its place over the median of the other floors at least two runs called present, and the densities behind that."""
    cells = row["cells"]
    d = {f: _density(paths[f], c["chk"], half) for f, c in cells.items() if c["chk"] is not None}
    ratios = {}
    for f in cells:
        others = [d[g] for g, c in cells.items() if g != f and g in d and c["v"] >= 2]
        ratios[f] = round(d[f] / (float(np.median(others)) + 1e-6), 3) if f in d and others else None
    return ratios, {f: round(v, 4) for f, v in d.items()}


def _cut(path, x, y):
    im = cv2.imread(str(path))
    r = SIDE // 2
    padded = cv2.copyMakeBorder(im, r, r, r, r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    x, y = min(max(x, 0), im.shape[1]), min(max(y, 0), im.shape[0])   # a registered place can land off the sheet
    return cv2.resize(padded[y:y + SIDE, x:x + SIDE], None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)


def _ask(ctx, path, chk, name):
    """Saves the crop at a place and returns the blind answer, or None if the reply cannot be used. The question never mentions other floors or what the runs said."""
    x, y = int(round(chk[0] / SNAP)) * SNAP, int(round(chk[1] / SNAP)) * SNAP
    cv2.imwrite(str(ctx.dir / name), _cut(path, x, y), [cv2.IMWRITE_JPEG_QUALITY, 92])
    try:
        answer = spark(QUESTION, [ctx.dir / name], meter=ctx.meter, schema=ANS, effort="low", max_tokens=4000, fresh=ctx.fresh)["answer"]
    except BAD_REPLY as e:
        log.warning("A stair crop was not read: %s", type(e).__name__)   # the exception text can quote the reply
        return None
    return answer if answer in ("yes", "no", "unclear") else None


def _copies(ctx, order):
    """Each floor's 2000 px JPEG, the file the floor order call read, with its size and the matrix from sheet px to its px."""
    image = {s["index"]: s["image"] for s in ctx.sheets}
    paths, sizes, scales = {}, {}, {}
    for i in order:
        sheet = cv2.imread(str(ctx.dir / image[i]))
        copy = _resized(sheet, WORK_PX)
        paths[i] = ctx.dir / f"floor_{i}.jpg"
        cv2.imwrite(str(paths[i]), copy, [cv2.IMWRITE_JPEG_QUALITY, 92])
        sizes[i] = copy.shape[1], copy.shape[0]
        scales[i] = np.diag([copy.shape[1] / sheet.shape[1], copy.shape[0] / sheet.shape[0], 1.0])
    return paths, sizes, scales


def _claims(reply, run, order, sizes, H):
    """One dict per present stair point of one run, in 2000 px copy pixels. Shafts stop here. A wrong shape raises, and so does a run that places no stair, because strict decoding delivers a blank answer as empty lists."""
    points = []
    for e, it in enumerate(reply["stairs"]):
        if type(it["name"]) is not str:
            raise ValueError("The reply does not match the schema")
        for k, fl in enumerate(it["floors"]):
            if type(fl["present"]) is not bool or type(fl["x"]) is not int or type(fl["y"]) is not int:
                raise ValueError("The reply does not match the schema")
            f = floor_index(fl["floor"], k, len(order), order)
            if f is None or not fl["present"]:
                continue
            w, h = sizes[f]
            raw = (min(max(fl["x"], 0), 1000) * w / 1000, min(max(fl["y"], 0), 1000) * h / 1000)
            points.append(dict(run=run, ent=f"{run}/{e}", name=it["name"], floor=f, raw=raw, ref=np.array(warp_point(H[f], *raw)) if f in H else None))
    if not points:
        raise ValueError("The run places no stair")
    return points


def _compare(ctx, order, paths, sizes, H, labels):
    """(the present points of every usable run, how many runs were not usable). The three requests are identical, so `run` keeps them three calls."""
    prompt = FLOOR_P.format(order=", ".join(f"F{k + 1} ({labels.get(i, '')})" for k, i in enumerate(order))) + EXTRA
    images = [paths[i] for i in order]

    def one(run):
        try:
            return _claims(spark(prompt, images, meter=ctx.meter, schema=SCHEMA, effort="medium", fresh=ctx.fresh, run=run), run, order, sizes, H)
        except BAD_REPLY as e:
            log.warning("Comparison run %d was not read: %s", run, type(e).__name__)
            return e

    with ThreadPoolExecutor(RUNS) as pool:
        runs = list(pool.map(one, range(RUNS)))
    usable = [run for run in runs if isinstance(run, list)]
    if not usable:
        for e in runs:
            if isinstance(e, APIError) and not isinstance(e, BadRequestError):   # a finished stage is never rerun, so fail it and let the next run retry, except on a 400, which a resend gets again
                raise e
    return [p for run in usable for p in run], RUNS - len(usable)


def _row_id(n):
    """A to Z, then AA, AB and on."""
    return _row_id(n // 26 - 1) + chr(65 + n % 26) if n >= 26 else chr(65 + n)


def _rows(points, order, H):
    """Rows with per-floor votes. chk is the marker and where local checks look: the registered place, because Spark's point drifts up to 112 px, or Spark's median on a floor with no registered place."""
    rows = []
    for n, c in enumerate(cluster(points)):
        cells = {}
        for f in order:
            here = [p for p in c["pts"] if p["floor"] == f]
            chk = warp_point(np.linalg.inv(H[f]), *c["pos"]) if c["pos"] is not None and f in H else None
            if chk is None and here:
                chk = tuple(float(v) for v in np.median([p["raw"] for p in here], axis=0))
            cells[f] = dict(v=len({p["run"] for p in here}), chk=chk, aligned=f in H, crop=None, file=None, mask=None, box=None)
        rows.append(dict(id=_row_id(n), name=Counter(p["name"] for p in c["pts"]).most_common(1)[0][0], pos=c["pos"], cells=cells, established=max(x["v"] for x in cells.values()) >= 2))
    return rows


def _evidence(ctx, rows, paths):
    """Ink and blank on every cell, then a crop on every cell of an established row and on cells a run listed in other rows, unless the place is blank. Returns the crops with no usable reply."""
    (ctx.dir / "verticals").mkdir(exist_ok=True)
    jobs = []
    for r in rows:
        r["ink"] = ink_ratio(r, paths, INK_HALF)[0]
        density = ink_ratio(r, paths, BLANK_HALF)[1]
        for f, c in r["cells"].items():
            c["blank"] = c["chk"] is None or density.get(f, 0) < BLANK
            if (r["established"] or c["v"] >= 1) and not c["blank"]:
                c["file"] = f"verticals/{r['id']}_{f}.jpg"
                jobs.append((f, c))
    with ThreadPoolExecutor(WORKERS) as pool:
        for done, ((_, c), answer) in enumerate(zip(jobs, pool.map(lambda job: _ask(ctx, paths[job[0]], job[1]["chk"], job[1]["file"]), jobs)), 1):
            c["crop"] = answer
            ctx.progress(f"Checking stair crop {done} of {len(jobs)}", done, len(jobs))
    return sum(c["crop"] is None for _, c in jobs)


def _decide_row(r, order):
    for f, c in r["cells"].items():
        c["state"], c["why"] = decide({"established": r["established"], "ink": r["ink"].get(f)}, c)
    if r["established"]:   # at least one cell has two runs, and such a cell is found or check, so the span exists
        held = [k for k, f in enumerate(order) if r["cells"][f]["state"] in ("found", "check")]
        for k, f in enumerate(order):
            if r["cells"][f]["state"] == "missing" and not held[0] <= k <= held[-1]:
                r["cells"][f]["state"] = "end"


def _zoom(im, box, k, minside):
    """The window of side k times the box, at least minside, enlarged toward 1000 px. Returns the image and the window."""
    w, h = box[2] - box[0], box[3] - box[1]
    side_w, side_h = max(k * w, minside), max(k * h, minside)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x1, y1 = max(0, int(cx - side_w / 2)), max(0, int(cy - side_h / 2))
    x2, y2 = min(im.shape[1], int(cx + side_w / 2)), min(im.shape[0], int(cy + side_h / 2))
    c = im[y1:y2, x1:x2]
    z = min(5.0, 1000 / max(c.shape[:2]))
    return cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA), (x1, y1, x2, y2)


def _whole(ctx, path, place):
    """A box tight around the whole stair at this place on one 2000 px copy, or None. The lab started from a loose Spark box, which does not exist here, so the first window is its smallest side at the registered place."""
    im = cv2.imread(str(path))
    cur, found = [*place, *place], None
    for k, minside in SEED_ROUNDS:
        crop, (x1, y1, x2, y2) = _zoom(im, cur, k, minside)
        # The lab left effort unset and the model chose about 2300 output tokens a call. `spark` always sends an effort, and medium measured about 1900.
        objects = spark(WHOLE_P, [crop], meter=ctx.meter, schema=GROUND, effort="medium", fresh=ctx.fresh)["objects"]
        if not objects:
            break
        w, h = x2 - x1, y2 - y1
        boxes = [[o["x_min"], o["y_min"], o["x_max"], o["y_max"]] for o in objects]
        boxes = [[min(a, c) * w / 1000 + x1, min(b, d) * h / 1000 + y1, max(a, c) * w / 1000 + x1, max(b, d) * h / 1000 + y1] for a, b, c, d in boxes]
        cx, cy = (cur[0] + cur[2]) / 2, (cur[1] + cur[3]) / 2
        cur = found = min(boxes, key=lambda b: np.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy))
    return found


def _outline(mask, left, top):
    """The largest blob of a box-local mask as frame pixels, or None for a mask too small to outline."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea, default=None)
    return None if c is None or len(c) < 3 else c.reshape(-1, 2) + (left, top)


def _box_dist(p, b):
    """Pixels from a point to a box, 0 inside."""
    return float(np.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3])))


def _draw(ctx, rows, order, floors, paths, sizes, scales, H):
    """Outlines Found cells: one SAM object per stair, pinned with one box on every Found floor's frame. Any failure leaves `mask` null, because the matrix is already decided. Returns the stairs left with no outline, and the spend cap's message if the cap stopped the outlines."""
    targets = [r for r in rows if any(c["state"] == "found" for c in r["cells"].values())]
    rw, rh = sizes[floors["reference"]]
    fw, fh = floors["frame_size"]
    stop = None

    def seed(r):
        """(row, its Found floors, its box in stack px), or None."""
        nonlocal stop
        found = [f for f in order if r["cells"][f]["state"] == "found"]
        f = found[(len(found) - 1) // 2]
        try:
            box = _whole(ctx, paths[f], r["cells"][f]["chk"])
            if box is None:
                return None
            corners = [warp_point(H[f], x, y) for x in box[::2] for y in box[1::2]]
            xs, ys = [x * fw / rw for x, _ in corners], [y * fh / rh for _, y in corners]
            x1, y1, x2, y2 = max(0, round(min(xs))), max(0, round(min(ys))), round(max(xs)), round(max(ys))
            x2, y2 = min(fw - 1, max(x2, x1 + 2)), min(fh - 1, max(y2, y1 + 2))
            return (r, found, (x1, y1, x2, y2)) if x1 < x2 and y1 < y2 else None
        except BudgetReached as e:
            stop = str(e)
            return None
        except Exception as e:
            log.warning("Stair %s has no outline seed: %s", r["id"], type(e).__name__)
            return None

    ctx.progress("Drawing the stairs")
    with ThreadPoolExecutor(WORKERS) as pool:
        seeds = [s for s in pool.map(seed, targets) if s]
    for start in range(0, len(seeds), SAM_OBJECTS):
        part = dict(enumerate(seeds[start:start + SAM_OBJECTS], start + 1))
        frames = {}
        for o, (_, found, (x1, y1, x2, y2)) in part.items():
            for f in found:
                frames.setdefault(order.index(f), []).append(f"{o}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={fw};h={fh}|>")
        prompt = "".join(f"<{k}f>" + ",".join(boxes) for k, boxes in sorted(frames.items()))
        try:
            for rec in sam_video(prompt, ctx.dir / floors["stack"], len(order), meter=ctx.meter):
                if rec["object"] not in part or rec["frame"] >= len(order) or (rec["conf"] or 0) < MASK_CONF:
                    continue
                r = part[rec["object"]][0]
                f = order[rec["frame"]]
                c = r["cells"][f]
                x1, y1, x2, y2 = rec["box"]
                outline = _outline(rec["mask"], x1, y1)
                if c["state"] != "found" or outline is None or _box_dist(r["pos"], (x1 * rw / fw, y1 * rh / fh, x2 * rw / fw, y2 * rh / fh)) > MASK_REACH:
                    continue
                to_sheet = np.linalg.inv(scales[f]) @ np.linalg.inv(H[f]) @ np.diag([rw / fw, rh / fh, 1.0])
                c["mask"] = [[round(x, 1), round(y, 1)] for x, y in (warp_point(to_sheet, *p) for p in outline)]
                xs, ys = zip(*c["mask"])
                c["box"] = [min(xs), min(ys), max(xs), max(ys)]
        except BudgetReached as e:
            stop = str(e)
        except Exception as e:
            log.warning("A stair outline request failed: %s", type(e).__name__)
    return sum(not any(c["mask"] for c in r["cells"].values()) for r in targets), stop


def _sheet_px(p, scale):
    return None if p is None else [round(float(p[0] / scale[0, 0]), 1), round(float(p[1] / scale[1, 1]), 1)]


def verticals_stage(ctx) -> dict:
    """`verticals` is one row per place and `not_checked` lists what was left out. Points, boxes and masks are in that floor's sheet pixels, `pos` in the reference sheet's, and `crop` is a path inside the review folder."""
    floors = ctx.stage("floors")
    order = floors.get("order") or []
    notes = [SHAFTS]
    if len(order) < 2:
        return {"verticals": [], "not_checked": notes + [floors.get("not_checked") or "Stairs were not followed, because the floor stage did not finish."]}
    paths, sizes, scales = _copies(ctx, order)
    H = {int(i): scales[floors["reference"]] @ np.array(T, float) @ np.linalg.inv(scales[int(i)]) for i, T in floors["transforms"].items()}   # floor copy px to reference copy px
    ctx.progress("Comparing the floors")
    labels = {s["index"]: s.get("level", "") for s in ctx.stage("read").get("sheets", [])}
    points, failed = _compare(ctx, order, paths, sizes, H, labels)
    if failed == RUNS:
        return {"verticals": [], "not_checked": notes + ["No stair comparison run placed a stair or gave a usable reply, so stairs were not followed."]}
    if failed:
        notes.append(f"{failed} of the 3 stair comparison runs gave no usable reply, so the stair matrix uses the others.")
    rows = _rows(points, order, H)
    unread = _evidence(ctx, rows, paths)
    if unread:
        notes.append(f"Stair crops with no usable reply: {unread}. A cell with no crop reply shows as Check.")
    for r in rows:
        _decide_row(r, order)
    bare, stop = _draw(ctx, rows, order, floors, paths, sizes, scales, H)
    if stop:
        notes.append(f"{stop} The stair outlines stopped there, and the matrix is not affected.")
    if bare:
        notes.append(f"Stairs with no outline drawn: {bare}. An outline is a picture only, so their cells are unchanged.")
    return {"verticals": [{"id": r["id"], "name": r["name"], "established": r["established"], "pos": _sheet_px(r["pos"], scales[floors["reference"]]),
                           "floors": {str(f): {"state": c["state"], "signals": {"spark": f"{c['v']}/3", "crop": c["crop"], "ink": r["ink"].get(f), "aligned": c["aligned"], "reason": c["why"]},
                                               "point": _sheet_px(c["chk"], scales[f]), "box": c["box"], "crop": c["file"], "mask": c["mask"]} for f, c in r["cells"].items()}} for r in rows],
            "not_checked": notes}
