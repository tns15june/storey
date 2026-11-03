"""The second review: each earlier comment at its place on the new set, and what changed that no comment covers.

Code sets every state from a registered pixel difference and the checks' own result on the new set. Spark only lists revision clouds and deltas, as a draft for the examiner that tells the changes the applicant marked from new ones, so a cloud never decides a state, and SAM takes no part.
Sheets are compared on their 2000 px copies, the size the threshold was measured on, whatever size ingest wrote, and each comment's box again at the sheet's own size."""
import itertools
import logging
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import cv2
import numpy as np

from .assemble import MOVED_AGAIN, _cut
from .declared import sheet_key
from .findings import CHECKS, DWELLING, RULE_BASED, STAGES as FACT_STAGES, _union
from .floors import ALIGN_SECONDS, GATE, SCALE_GATE, WORK_PX, _resized, _sheet_name, register, warp_point
from .meta import BudgetReached, spark
from .verticals import BAD_REPLY

log = logging.getLogger(__name__)

# Set after seeing data, so Task 14 tests it again: lab/defects/rev_eval.py fitted this cut on dev buildings, and held-out sheets gave 21 of 30 changed places over it and 0 of 42 unchanged ones.
CHANGED = 0.51     # ink that differs, over the ink in both sheets. A wall added across a room scored under it in the lab, and a miss reads Still open, which is the safe direction
STROKE = 40        # grey levels, the lab's per-pixel cut in lab/samtricks/item7.py
MISFIT = 5         # px, the side of the square round a stroke that must hold no ink in the other sheet. On 27 unchanged real sheets through a 0.2% keystone: 20 false regions, against 140 at 3 px and 247 for a ratio over a sliding window. 3 px finds more changed digits on Capitol, 25 of 36 against 13, so Task 14 tests this too
FAINT = 2.0        # ink in a place with almost none, where one speck would be most of the ratio
JOIN = 30          # px, so the treads of one stair make one change and not one per tread
EDGE = 9           # px of the frame the new sheet does not reach for certain: a rescan pushes the border off the page, which gave 50 false regions on those 27 sheets without this
MAX_CHANGES = 40   # per sheet, because a redrawn sheet gives hundreds, and a list that long reads as one fact
# Set after seeing data, so Task 14 tests them again. On the 27 unchanged real sheets above, 2333 boxes of text-layer words and scan notes gave 0 changed boxes through the rescan, stretch and keystone.
# A one-digit swap on Capitol A2.1 to A2.3 reached GONE on 34 of 36, rescanned as well, and the smallest correction on the planned revision set gave 10. Without the half-strength rule, blur and JPEG gave 19 boxes over GONE.
NEAR = 3           # px at the sheet's own size, the side of the square round an old stroke that must hold no new ink
SLACK = 6          # px a box may shift to line up with the new sheet, which takes up what the sheet's fit leaves: with no shift the keystone gave 912 false boxes and the stretch 225, and with 2 px the keystone gave 235
AROUND = 100       # px of the sheet round a box that the shift lines up, so a box is not moved onto its own correction
GONE = 5           # px of old ink in a box that lost over half its strength
MIDDLE = 1 / 3     # share of each side of a stair's box, round its centre, where ink added counts. Set after seeing data, so Task 14 tests it again: on 710 A1.3 a cloud drawn round the stair put 4244 px of new ink in the box and none in this middle, where the stairs drawn back put 3453 there, and 4519 on St Elizabeths
MARK_MAX = 0.05   # share of the sheet a listed cloud or delta may cover and still mark a change. The largest of the lab's 20 real clouds covers 1.9%, and a model's box over a whole tile, a tenth of the sheet, would hide every change under it
WORKERS = 6
GRID = (4, 3)      # tiles along the long and the short edge, 12 in all
TILE_OVERLAP = 0.1
MAX_CLOUDS = 20    # per tile, because a reply is untrusted
DELTA_MAX = 12     # characters

CLOUDS_DRAFT = "The revision cloud list is a draft read by a model, and a sheet with no cloud listed can still have changed."
NOT_COMPARED = "The last review's checks did not finish, so the comments they would have raised were not compared."
AGAIN = "The checks raised this comment again."
NO_MATCH = "This sheet has no match in the new set."
NOT_CARRIED = "Some comments the last review held open were not compared, because its comparison did not finish or did not record them."
NOT_LINED_UP = "This sheet could not be lined up with the new set."
STILL = "This {} did not change."   # place, or for a comment with no box its sheet, or the whole set if it names no sheet
STILL_GONE = "This {} did not change, although the checks no longer raise this comment."
MOVED_GONE = "This {} changed, and the checks no longer raise this comment."
NOT_TOLD = "This {} changed, but the checks could not tell on the new set whether this comment still applies."
NEW_CHANGE = "Changed since the last set, and no comment covers it."
AT_COMMENT = "Changed since the last set, around comment {}."
MARKED = " A revision cloud or delta on the draft list marks it, so the applicant flagged this change."

RAISED_BY = {1: ("required sheets", "sheets not read", "plans outside the floor order"), 2: ("sheet index",), 3: ("sheet titles",), 4: ("code summary",), 5: ("stories against floor plans", "code edition"), 6: ("table arithmetic",),
             7: ("door and window tags", "door type against its swing", "window sizes", "swing column"), 8: ("exit capacity",), 9: ("stair continuity",), 10: ("stairs at the discharge level",), 11: ("exits per floor",), 13: ("travel distance",),
             14: ("corridor width",), 16: ("room names",), 17: ("dimension strings",)}   # CHECKS is keyed by label, and a finding holds only its number. A finding from an unfinished read has its check's number, and every such number's row already reads the declared stage
STAIR_CHECKS = (9, 10, 11)
FLOOR_CHECKS = (*STAIR_CHECKS, 13, 14)   # these look only at floors in the order, while the other checks read sheets such as a code sheet that no order holds

BOX = {k: {"type": "integer"} for k in ("x1", "y1", "x2", "y2")}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["clouds", "deltas"], "properties": {
    "clouds": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["x1", "y1", "x2", "y2", "delta"], "properties": {**BOX, "delta": {"type": "string"}}}},
    "deltas": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["x1", "y1", "x2", "y2", "number"], "properties": {**BOX, "number": {"type": "string"}}}}}}
PROMPT = ("This image is from a sheet of a building drawing set, part of a revised submittal. A revision cloud is a closed outline made of small scalloped arcs, drawn around a changed area. "
          "A delta is a small triangle holding a revision number, placed next to a cloud or in a revision list. List every revision cloud and every delta triangle in this image, however small. "
          "For each cloud give its box and the number in the nearest delta, or an empty string if there is none. For each delta give its box and the number inside it. "
          "Coordinates are integers on a 0-1000 grid of this image, x from the left edge and y from the top edge. Return empty lists if there are none.")


def _soft_ink(image):
    """Ink as a positive signal with flat paper removed, blurred by one pixel: the map `CHANGED` was measured on."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    paper = cv2.resize(cv2.medianBlur(cv2.resize(gray, None, fx=0.25, fy=0.25), 31), (gray.shape[1], gray.shape[0]))
    return cv2.GaussianBlur(np.clip(paper.astype(np.float32) - gray, 0, 255), (0, 0), 1)


def _changed(a, b, box):
    """The lab's ratio over one box of the two ink maps."""
    x1, y1, x2, y2 = (max(v, 0) for v in box)
    p, q = a[y1:y2, x1:x2], b[y1:y2, x1:x2]
    if not p.size:
        return False
    ink = p.mean() + q.mean()
    return bool(ink > FAINT and np.abs(p - q).mean() > CHANGED * ink)


def _gone(old, new, H, box, middle=1.0):
    """Pixels of old ink in a box, in old sheet px, that lost over half its strength with no new ink within a pixel, read at the sheet's own size, where the 2000 px copy blurs one digit into another.
    The box and the sheet round it take the shift that leaves least ink unmatched, nearest first. A stroke only dimmed by blur or JPEG keeps over half its ink, and a cloud drawn round the place adds ink and takes none away, so neither counts.
    Read with the sheets swapped, it counts ink added, a cloud's as well. Only the `middle` share of each side, round the box's centre, is counted."""
    # shortcut: no box measured lay within 60 px of the sheet's edge, where a rescan can push ink off the page, mask the new sheet's reach as `_fit` does if a comment there reads changed.
    x1, y1, x2, y2 = (int(v) for v in box)
    m = AROUND + SLACK
    w, h = x2 - x1 + 2 * m, y2 - y1 + 2 * m
    at = np.array([[1, 0, m - x1], [0, 1, m - y1], [0, 0, 1.0]])
    a = _soft_ink(cv2.warpPerspective(old, at, (w, h), borderValue=(255, 255, 255)))[SLACK:h - SLACK, SLACK:w - SLACK]
    b = cv2.dilate(_soft_ink(cv2.warpPerspective(new, at @ H, (w, h), borderValue=(255, 255, 255))), np.ones((NEAR, NEAR), np.uint8))
    cut = np.maximum(STROKE, a / 2)
    shifts = sorted(itertools.product(range(-SLACK, SLACK + 1), repeat=2), key=lambda s: s[0] ** 2 + s[1] ** 2)
    left = min((a - b[SLACK + dy:h - SLACK + dy, SLACK + dx:w - SLACK + dx] > cut for dx, dy in shifts), key=np.count_nonzero)
    ox, oy = int((x2 - x1) * (1 - middle) / 2), int((y2 - y1) * (1 - middle) / 2)
    return np.count_nonzero(left[AROUND + oy:AROUND + y2 - y1 - oy, AROUND + ox:AROUND + x2 - x1 - ox])


def _fit(old, new, deadline):
    """(matrix from new sheet px to old sheet px, both sheets' ink on the old sheet's 2000 px copy), or None if the pair cannot be compared.
    Registration reads whole sheets, because a pair is one sheet twice, and Spark's plan box can cover 13% of a sheet's width, which leaves the fit extrapolated across the rest: 6 to 10 false regions on four unchanged Capitol sheets.
    A wrong fit would make every comment read changed, so a sheet that differs across most of its ink is not compared either, however little ink it holds."""
    if old is None or new is None or time.monotonic() > deadline:
        return None
    H, inliers = register(old, new, [0, 0, old.shape[1], old.shape[0]], [0, 0, new.shape[1], new.shape[0]])
    if inliers < GATE or abs(np.hypot(H[0, 0], H[1, 0]) * max(new.shape[:2]) / max(old.shape[:2]) - 1) > SCALE_GATE:
        return None
    small, moved = _resized(old, WORK_PX), _resized(new, WORK_PX)
    h, w = small.shape[:2]
    to_old = np.diag([w / old.shape[1], h / old.shape[0], 1.0]) @ H @ np.diag([new.shape[1] / moved.shape[1], new.shape[0] / moved.shape[0], 1.0])
    warped = cv2.warpPerspective(moved, to_old, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    reach = cv2.warpPerspective(np.full(moved.shape[:2], 255, np.uint8), to_old, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    reach = cv2.erode(reach, np.ones((EDGE, EDGE), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    a, b = _soft_ink(small), _soft_ink(warped)
    a[~reach] = b[~reach] = 0
    return None if np.abs(a - b).mean() > CHANGED * (a.mean() + b.mean()) else (H, a, b)


def _regions(a, b):
    """Boxes round strokes in one map with no ink near them in the other, on the 2000 px copy, so a line that moved a pixel or two is no change, and a word gone from a busy sheet is one."""
    near = np.ones((MISFIT, MISFIT), np.uint8)
    strokes = np.maximum(a - cv2.dilate(b, near), b - cv2.dilate(a, near)) > STROKE
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(np.uint8(strokes), np.ones((JOIN, JOIN), np.uint8)))
    return [[x, y, x + w, y + h] for x, y, w, h, _ in stats[1:].tolist()]


def _mostly_inside(region, box):
    w, h = min(region[2], box[2]) - max(region[0], box[0]), min(region[3], box[3]) - max(region[1], box[1])
    return w > 0 and h > 0 and 2 * w * h >= (region[2] - region[0]) * (region[3] - region[1])


def _mapped(H, box):
    """A box through a matrix, as the box round its corners."""
    xs, ys = zip(*(warp_point(H, x, y) for x in box[::2] for y in box[1::2]))
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def _row(f, state, why, old_crop=None, new_crop=None):
    """`f` is the comment as it stands on the new set, its sheet and box in that set's sheet pixels, and the row keeps it so the next cycle compares it again while it is not resolved."""
    return {"comment_id": f["comment_id"], "title": f["title"], "state": state, "old_crop": old_crop, "new_crop": new_crop, "why": why, "comment": f}


def _unplaced(f, again, proven, changed, where):
    """A comment with no box, read as one whose place is its sheet, or the whole set (`where`), so the checks dropping it while nothing changed still reads Still open."""
    # shortcut: a sheet whose only change is a drawn cloud counts as changed here, so such a comment can read Looks resolved once the checks drop it, give its finding a box if a real set shows this.
    return _row(f, "open", AGAIN) if again else _row(f, *_verdict(changed, False, proven, where))


def _checked(f, sheet, stored, evaluated):
    """True only on proof that the checks compared this comment's own item on the new set with every fact it rests on read, which evaluate gives as its key in `compared`.
    Without that, a comment the checks did not raise proves nothing. The stage, check and sheet doubts below are a second guard, and `sheet` is the new set's index."""
    entries = [entry for entry in CHECKS if entry[0] in RAISED_BY.get(f["check"], ())]
    # shortcut: keys must match exactly, so a stair that drifted into the next grid cell reads Cannot tell although it was found, match keys a cell apart as assign_ids does if real reviews show this often.
    if f["key"] not in evaluated.get("compared", ()) or not entries or any(not stored[name] or "failed" in stored[name] for _, _, needs in entries for name in needs):
        return False
    lines = evaluated["not_checked"]
    if any(f"The {label} check did not finish." in lines or (DWELLING in lines and label in RULE_BASED) for label, _, _ in entries):
        return False
    floors = stored["floors"]
    if f["check"] in STAIR_CHECKS and not any(row.get("established") for row in stored["verticals"].get("verticals", ())):
        return False
    if f["check"] in FLOOR_CHECKS and (sheet not in (floors.get("order") or ()) or sheet in (floors.get("unaligned") or ())):
        return False
    return not any(g["check"] == f["check"] and g["severity"] == "unknown" and g["sheet"] in (None, sheet) for g in evaluated["findings"])   # a corridor that could not be measured leaves its comment unraised, under a key of its own


def _verdict(moved, again, proven, where="place"):
    """A changed place reads resolved only if the checks ran and agree. Each disagreement goes to Still open, the side where an examiner looks again, and a changed place with no proof goes to Cannot tell."""
    if again:
        state, why = "open", MOVED_AGAIN if moved else STILL
    elif not proven:
        state, why = ("unknown", NOT_TOLD) if moved else ("open", STILL)
    else:
        state, why = ("resolved_looks", MOVED_GONE) if moved else ("open", STILL_GONE)
    return state, why.format(where)


def _saved(image, box, path):
    return path.name if _cut(image, box, path) else None


def _compare(ctx, comments, raised, proven, old, new, fit, sheet):
    """(comment rows, new change rows, whether the sheet changed at all) for one matched pair. `proven` holds the comments `_checked` passed. `fit` is None for sheets with identical bytes, which hold no change."""
    H, a, b = fit or (np.eye(3), None, None)
    back = np.linalg.inv(H)
    kx, ky = (a.shape[1] / old.shape[1], a.shape[0] / old.shape[0]) if a is not None else (1, 1)
    regions = _regions(a, b) if a is not None else []
    rows, placed, held = [], {}, []
    for f in comments:
        if not f["box"]:
            rows.append(_unplaced(f | {"sheet": sheet}, f["comment_id"] in raised, f["comment_id"] in proven, bool(regions), "sheet"))
            continue
        name, (x1, y1, x2, y2) = f["comment_id"], f["box"]
        placed[name] = box = [int(x1 * kx), int(y1 * ky), int(x2 * kx) + 1, int(y2 * ky) + 1]
        at = _mapped(back, f["box"])   # the place on the new sheet
        # `CHANGED` was measured on erased and moved items, and a word replaced in place keeps most of its ink, 0.03 to 0.17 on the planned revision set, so ink gone at the sheet's own size counts too.
        # A stair drawn back adds ink and takes none away, so a stair's place also reads the new sheet against the old. Only there, because a cloud drawn round any item adds ink too: on 710 A1.2 a cloud alone put 2595 px of new ink in the floor name's box.
        # shortcut: a cloud round a stair narrower than the middle third of its box runs inside that third and counts, so a stair comment the checks drop with no stair drawn can read Looks resolved, tell scallops from treads if a real set shows it.
        moved = a is not None and (_changed(a, b, box) or _gone(old, new, H, f["box"]) >= GONE or f["check"] in STAIR_CHECKS and _gone(new, old, back, at, MIDDLE) >= GONE)
        if moved:
            held.append(box)
        rows.append(_row(f | {"sheet": sheet, "box": at}, *_verdict(moved, name in raised, name in proven), _saved(old, f["box"], ctx.dir / f"old_{name}.jpg"), _saved(new, at, ctx.dir / f"new_{name}.jpg")))
    changes = []
    for region in regions:
        if not any(_mostly_inside(region, box) for box in held):   # a changed place's row accounts for what it holds, and every other region stays listed, so no change found is lost
            names = [name for name, box in placed.items() if _touch(region, box)]
            changes.append((region, AT_COMMENT.format(", ".join(names)) if names else NEW_CHANGE))
    to_new = back @ np.diag([1 / kx, 1 / ky, 1.0])
    return rows, [{"sheet": sheet, "box": _mapped(to_new, r), "note": note, "state": "new_change"} for r, note in changes], bool(regions)


def _tiles(w, h):
    """Overlapping tiles as (x1, y1, x2, y2) in sheet pixels, 12 in all, long side along the long edge."""
    cols, rows = GRID if w >= h else GRID[::-1]
    tw, th = w / cols * (1 + TILE_OVERLAP), h / rows * (1 + TILE_OVERLAP)
    return [(float(x), float(y), float(x + tw), float(y + th)) for x in np.linspace(0, w - tw, cols) for y in np.linspace(0, h - th, rows)]


def _boxes(items, tile):
    """(item, box in sheet pixels) for each item on the tile's 0-1000 grid, leaving out boxes that enclose nothing."""
    left, top, right, bottom = tile
    for item in items[:MAX_CLOUDS]:
        x1, y1, x2, y2 = (min(max(int(item[k]), 0), 1000) / 1000 for k in ("x1", "y1", "x2", "y2"))
        if x1 < x2 and y1 < y2:
            yield item, [round(v, 1) for v in (left + x1 * (right - left), top + y1 * (bottom - top), left + x2 * (right - left), top + y2 * (bottom - top))]


def _clouds_in(reply, tile):
    """One tile's (clouds, delta boxes) in sheet pixels. A reply is untrusted, so a wrong shape raises and the tile counts as unread."""
    clouds = [{"box": box, "delta": str(cloud["delta"]).strip()[:DELTA_MAX] or None} for cloud, box in _boxes(reply["clouds"], tile)]
    return clouds, [box for _, box in _boxes(reply["deltas"], tile)]


def _touch(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _merged(rows):
    """Boxes that touch become one, which drops the repeats across tile overlaps, as the lab's nms did, and joins a cloud a tile edge cut in two."""
    # shortcut: a false box the size of a tile takes in the clouds it touches, which is harmless in a draft list, join only boxes that reach a tile edge if a real sheet shows it.
    rows = [dict(row) for row in rows]
    joined = True
    while joined:
        joined = False
        for a, b in itertools.combinations(rows, 2):
            if _touch(a["box"], b["box"]):
                a |= {"box": _union([a["box"], b["box"]]), "delta": a["delta"] or b["delta"]}
                rows.remove(b)
                joined = True
                break
    return rows


def _read_clouds(ctx, sheets):
    """(cloud rows, marks, lines for what was not read). Marks are the clouds and deltas small enough to say the applicant flagged a change there.
    A budget stop or a bad reply costs tiles and never the comment states, which are already set. One image at a time, because a set can hold many 6000 px sheets."""
    rows, marks, unread, total, stop = [], [], 0, 0, None

    def one(image, tile):
        nonlocal stop
        x1, y1, x2, y2 = (int(v) for v in tile)
        try:
            return _clouds_in(spark(PROMPT, [_resized(image[y1:y2, x1:x2], WORK_PX)], meter=ctx.meter, schema=SCHEMA, effort="low", fresh=ctx.fresh), tile)
        except BudgetReached as e:
            stop = str(e)
        except (*BAD_REPLY, OverflowError) as e:   # JSON reads 1e999 as infinity, and int() of that overflows
            log.warning("A cloud tile was not read: %s", type(e).__name__)   # the exception text can quote the reply

    for done, sheet in enumerate(sheets, 1):
        ctx.progress(f"Listing revision clouds on sheet {done} of {len(sheets)}", done, len(sheets))
        tiles = _tiles(sheet["width"], sheet["height"])
        image = cv2.imread(str(ctx.dir / sheet["image"]))
        with ThreadPoolExecutor(WORKERS) as pool:
            found = list(pool.map(lambda tile: one(image, tile), tiles))
        total += len(tiles)
        unread += sum(r is None for r in found)
        clouds = _merged(c for r in found if r for c in r[0])
        rows += [c | {"sheet": sheet["index"]} for c in clouds]
        boxes = [c["box"] for c in clouds] + [box for r in found if r for box in r[1]]
        marks += [{"sheet": sheet["index"], "box": box} for box in boxes if (box[2] - box[0]) * (box[3] - box[1]) <= MARK_MAX * sheet["width"] * sheet["height"]]
    lines = [f"{stop} The revision cloud list stopped there, and no comment state depends on it."] if stop else []
    return rows, marks, lines + ([f"Revision clouds were not read on {unread} of {total} tiles, so the cloud list is not complete."] if unread else [])


def _numbered(rows):
    """Read rows by sheet number. A number two sheets share names neither, so it matches nothing."""
    seen = Counter(sheet_key(r["number"]) for r in rows if r.get("read_ok") and r["number"])
    return {sheet_key(r["number"]): r for r in rows if seen[sheet_key(r["number"])] == 1}


def recheck_stage(ctx) -> dict:
    """{} on a first review. Otherwise `comments`, a row for each comment the last review raised and each it held open, as stored or as shown, that its checks no longer raised, `changes` and `clouds`, each with a `sheet` and `box`
    in the new sheet's pixels where it has one, `unmatched_sheets`, and `not_checked` lines. A row's `comment` is the comment as it stands on the new set. Crops are file names in the review folder, and None with no place to cut.
    A change reads `new_change`, or has no state where a listed cloud or delta marks it as the applicant's own."""
    # shortcut: `raised` is read under the first run's choices, so a later change of jurisdiction does not reset a state, recompute it in assemble if an examiner relies on that.
    from .pipeline import read_json   # here, because pipeline imports this module to register the stage

    parent = ctx.stage("upload").get("parent")
    if not parent:
        return {}
    before = replace(ctx, dir=ctx.dir.parent / parent)
    evaluated, review = before.stage("evaluate"), read_json(before.dir / "review.json")
    dropped = {key for key, decision in review.get("decisions", {}).items() if decision == "drop"}
    shown = {c["comment_id"]: c["state"] for c in (review.get("recheck") or {}).get("comments", ())}   # after a selection the screen can show a stored Looks resolved as Cannot tell, and the examiner saw that
    found, last = evaluated.get("findings", ()), before.stage("recheck")
    ids = {f["comment_id"] for f in found}
    outstanding = [c for c in last.get("comments", ()) if {c["state"], shown.get(c["comment_id"])} & {"open", "unknown"} and c["comment_id"] not in ids]   # the parent's checks no longer raise these, and nothing resolved them
    carried = [c["comment"] for c in outstanding if "comment" in c]
    comments = [f for f in (*found, *carried) if f["key"] not in dropped]
    latest, stored = ctx.stage("evaluate"), {name: ctx.stage(name) for name in FACT_STAGES}
    shared = {f["key"].partition("#")[0] for f in (*found, *carried) if "#" in f["key"]}   # the last set held two items under this key, so the new set's evidence for it can be either one's
    latest = latest | {"compared": [key for key in latest.get("compared", ()) if key not in shared]}
    raised = {f["comment_id"] for f in latest.get("findings", ())}
    old_rows, new_rows = before.stage("read").get("sheets", []), ctx.stage("read").get("sheets", [])
    old_by, new_by = _numbered(old_rows), _numbered(new_rows)
    numbers = [number for number in old_by if number in new_by]
    pairs = [(old_by[number], new_by[number]) for number in numbers]
    old_files = {s["index"]: s["image"] for s in read_json(before.dir / "sheets.json") or ()}
    new_sheets = {s["index"]: s for s in ctx.sheets}
    paired_old, paired_new = {o["index"] for o, _ in pairs}, {n["index"] for _, n in pairs}
    unmatched = [_sheet_name(r) for r in old_rows if r["index"] not in paired_old] + [_sheet_name(r) for r in new_rows if r["index"] not in paired_new]
    by_sheet = defaultdict(list)
    for f in comments:
        by_sheet[f["sheet"]].append(f)
    rows, changes = {}, []
    to_read = [new_sheets[r["index"]] for r in new_rows if r["index"] not in paired_new]
    deadline = time.monotonic() + ALIGN_SECONDS
    for done, (o, n) in enumerate(pairs, 1):
        ctx.progress(f"Comparing sheet {done} of {len(pairs)}", done, len(pairs))
        here = by_sheet.pop(o["index"], [])
        old_path, new_path = before.dir / old_files[o["index"]], ctx.dir / new_sheets[n["index"]]["image"]
        same = old_path.read_bytes() == new_path.read_bytes()
        if same and not here:
            continue
        old, new = cv2.imread(str(old_path)), cv2.imread(str(new_path))
        fit = None if same else _fit(old, new, deadline)
        if not same and fit is None:
            unmatched.append(_sheet_name(n))
            to_read.append(new_sheets[n["index"]])
            rows |= {f["comment_id"]: _row(f | {"sheet": n["index"], "box": None}, "unknown", NOT_LINED_UP) for f in here}
            continue
        proven = {f["comment_id"] for f in here if _checked(f, n["index"], stored, latest)}
        compared, new_changes, differs = _compare(ctx, here, raised, proven, old, new, fit, n["index"])
        rows |= {row["comment_id"]: row for row in compared}
        changes += new_changes
        if differs:
            to_read.append(new_sheets[n["index"]])
    for sheet, here in by_sheet.items():
        for f in here:   # the set changed if a sheet came, went, could not be lined up or differs, and each of those is in `to_read` or `unmatched`
            # shortcut: a comment with no place on the new set goes on as one on its sheet, or on the whole set where its sheet has no match, so a later cycle cuts no crop for it and takes any change there as its place changing, keep the last set that showed it if a real review needs this.
            rows[f["comment_id"]] = _unplaced(f, f["comment_id"] in raised, _checked(f, None, stored, latest), bool(to_read or unmatched), "set") if sheet is None else _row(f | {"sheet": None, "box": None}, "unknown", NO_MATCH)
    clouds, marks, lines = _read_clouds(ctx, to_read)
    marks = [m for m in marks if any(c["sheet"] == m["sheet"] and _mostly_inside(m["box"], c["box"]) for c in changes)]   # this revision's marks are new ink, and a cloud or delta left from an earlier one is on both sets, so no change holds it
    for change in changes:
        if any(m["sheet"] == change["sheet"] and _touch(change["box"], m["box"]) for m in marks):
            change |= {"note": change["note"] + MARKED, "state": None}
    for sheet in {c["sheet"] for c in changes}:   # after marking, so a redrawn sheet's one box never reads as flagged by the applicant
        many = [c for c in changes if c["sheet"] == sheet and c["state"]]
        if len(many) > MAX_CHANGES:
            changes = [c for c in changes if c not in many] + [{"sheet": sheet, "box": _union(c["box"] for c in many), "note": f"Changed in {len(many)} places since the last set, shown as one box.", "state": "new_change"}]
    return {"comments": [rows[f["comment_id"]] for f in comments], "changes": changes, "clouds": clouds, "unmatched_sheets": list(dict.fromkeys(unmatched)),
            "not_checked": [CLOUDS_DRAFT, *lines, *([] if "findings" in evaluated else [NOT_COMPARED]),
                            *([NOT_CARRIED] if before.stage("upload").get("parent") and ("comments" not in last or len(carried) < len(outstanding)) else [])]}
