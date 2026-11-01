"""The test tool: removes one stair from a copy of a sheet, so a run of the stair stage can show the check is live. The original review is only ever read.

The removal follows the soft plant of lab/audit2/plants.py and lab/defects/plant.py, because that is what the stair rule was tested on: the stair goes and the walls stay, so the room is still drawn.
It runs the rule's ink test on its own result before any credit is spent. On a set like habs_stelizabeths_center Spark claims an erased stair on all three runs, so a leftover well and tread stubs read Found."""
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import pipeline
from .assemble import JPEG
from .findings import _cell_box
from .floors import WORK_PX, _resized, _sheet_name, ink
from .recheck import _soft_ink
from .verticals import INK_LOW, ink_ratio

THICK = 5   # px on a 2000 px sheet: a stroke this wide survives, so walls stay and treads go
WALL_TONE, WALL_WIDE, WALL_LONG = 25, 11, 30   # grey levels and px. The ink map sees a hatched wall as two thin outlines, so the lab's rule erased it (share removed 1.0 on habs_va_bldg1 east), and a grey band this long is a wall and not a room label or treads blurred together
SEED = 1
NOT_CLEAN = "This stair could not be removed cleanly on this sheet."
REUSED = ("stage_read.json", "stage_floors.json", "stage_declared.json", "stage_vector.json", "stack.mp4", "sheets.json", "decisions.json")   # everything else is rebuilt, so nothing unedited sits beside the new finding


def _erase(img, box, region):
    """Removes ink inside the bool `region`. Paper tone comes from a normalised blur over nearby paper, and grain from real paper residuals. Pixels away from ink stay byte for byte."""
    x1, y1, x2, y2 = box
    pad = 39
    X1, Y1, X2, Y2 = max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, img.shape[1]), min(y2 + pad, img.shape[0])
    c = img[Y1:Y2, X1:X2].astype(np.float32)
    local = cv2.GaussianBlur(cv2.morphologyEx(img[Y1:Y2, X1:X2], cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8)), (0, 0), 8).astype(np.float32)
    dark = (local - c).mean(axis=2)
    paper = np.float32(cv2.erode(np.uint8(dark < 10), np.ones((5, 5), np.uint8)))
    if paper.sum() < 100:
        paper = np.float32(dark < 25)
    num = cv2.GaussianBlur(c * paper[..., None], (0, 0), 14)
    den = cv2.GaussianBlur(paper, (0, 0), 14)[..., None] + 1e-3
    weight = den / (den + 0.12)   # in dense ink there is no nearby paper, so lean on the sheet's own paper tone
    tone = weight * num / den + (1 - weight) * np.median(img.reshape(-1, 3), axis=0).astype(np.float32)
    grain = (c - tone)[paper > 0]
    grain = grain[np.abs(grain).max(axis=1) < np.percentile(np.abs(grain).max(axis=1), 98) + 1e-3]
    mask = np.uint8(region[Y1:Y2, X1:X2])
    strokes = cv2.dilate(np.uint8(((tone - c).mean(axis=2) > 6) & (mask > 0)), np.ones((3, 3), np.uint8)) & cv2.dilate(mask, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(np.float32(strokes), (0, 0), 0.7)[..., None]
    fill = tone + grain[np.random.default_rng(SEED).integers(0, len(grain), c.shape[:2])]
    out = img.copy()
    out[Y1:Y2, X1:X2] = np.clip(c * (1 - alpha) + fill * alpha, 0, 255).astype(np.uint8)
    return out


def _soft(im, box):
    """Erases everything inside a box but its walls: thick ink and long grey bands that run on past the box. A well or a block of treads ends inside it."""
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(x1 + 2, 0), max(y1 + 2, 0), x2 - 2, y2 - 2
    grey = np.uint8(_soft_ink(im) > WALL_TONE)
    thick = cv2.morphologyEx(np.uint8(ink(im) > 0), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (THICK, THICK)))
    for band in ((WALL_WIDE, WALL_LONG), (WALL_LONG, WALL_WIDE)):
        thick |= cv2.morphologyEx(grey, cv2.MORPH_OPEN, np.ones(band, np.uint8))
    region = np.zeros(thick.shape, bool)
    region[y1:y2, x1:x2] = True
    labels = cv2.connectedComponents(thick)[1]
    walls = np.isin(labels, np.setdiff1d(labels[~region], 0))
    region &= cv2.dilate(np.uint8(walls), np.ones((5, 5), np.uint8)) == 0
    return _erase(im, (x1, y1, x2, y2), region)


def _remove(path, box):
    """Takes the stair off a sheet image. The plant's constants are pixels of a 2000 px sheet, so it runs on that copy, and its result goes on the full sheet only where a pixel moved, which keeps every other pixel."""
    sheet = cv2.imread(str(path))
    small = _resized(sheet, WORK_PX)
    h, w = sheet.shape[:2]
    kx, ky = small.shape[1] / w, small.shape[0] / h
    edited = _soft(small, (box[0] * kx, box[1] * ky, box[2] * kx, box[3] * ky))
    moved = cv2.resize(cv2.dilate(np.uint8((edited != small).any(axis=2)), np.ones((3, 3), np.uint8)), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    sheet[moved] = cv2.resize(edited, (w, h), interpolation=cv2.INTER_LINEAR)[moved]
    cv2.imwrite(str(path), sheet)


def _stair_box(cell):
    """The box round a stair, mirrored about its place, because SAM's outline can stop at the well: 38 px of a stair about 75 px wide on habs_stelizabeths_center."""
    x1, y1, x2, y2 = _cell_box(cell)
    px, py = cell["point"]
    half_x, half_y = max(px - x1, x2 - px), max(py - y1, y2 - py)
    return [px - half_x, py - half_y, px + half_x, py + half_y]


def _ink_left(dir, sheets, stair, sheet):
    """The stair rule's ink ratio at this place on the edited sheet, read as the rule reads it: from a 2000 px JPEG of each floor the stair is Found on."""
    paths, cells = {}, {}
    for s in sheets:
        i, cell = s["index"], stair["floors"].get(str(s["index"]))
        if i == sheet or (cell and cell["state"] == "found"):
            copy = _resized(cv2.imread(str(dir / s["image"])), WORK_PX)
            paths[i] = dir / f"floor_{i}.jpg"
            cv2.imwrite(str(paths[i]), copy, JPEG)
            cells[i] = {"chk": [v * copy.shape[1] / s["width"] for v in cell["point"]], "v": 2}
    return ink_ratio({"cells": cells}, paths)[0][sheet]


def _target(dir, vertical, sheet):
    """(the stair's row, the read row of its sheet) if a finding can appear there. A request names these, so each is checked against the review and neither reaches a path."""
    floors = pipeline.read_json(dir / "stage_floors.json")
    row = next((r for r in pipeline.read_json(dir / "stage_verticals.json").get("verticals", ()) if r["id"] == vertical), None)
    cell = row["floors"].get(str(sheet)) if row else None
    if cell is None or cell["state"] != "found":
        raise ValueError("That stair is not Found on that sheet.")
    order = floors["order"]
    here = order.index(sheet)
    found = [k for k, i in enumerate(order) if row["floors"].get(str(i), {}).get("state") == "found"]
    chosen = pipeline.read_json(dir / "stage_evaluate.json").get("choices", {}).get("discharge")
    level = floors.get("discharge") if chosen is None else chosen
    if not (found[0] < here < found[-1] or (sheet == level and found[-1] > here)):
        raise ValueError("A finding can appear only on a floor with the stair above and below it, or on the discharge level.")
    return row, next(r for r in pipeline.read_json(dir / "stage_read.json")["sheets"] if r["index"] == sheet)


def remove_stair(dir: Path, vertical: str, sheet: int) -> Path:
    """A new review folder holding a copy of this review with one stair removed, ready for `pipeline.run`, which runs verticals, evaluate, letter and assemble under the normal Meter. Raises ValueError for a stair or sheet this review cannot test, or a stair the removal cannot clear."""
    # shortcut: stack.mp4 keeps the unedited sheet, which only outlines on Found cells read, write the stack again if an outline ever appears on an edited floor.
    stair, row = _target(dir, vertical, sheet)
    upload = pipeline.read_json(dir / "stage_upload.json")
    sheets = pipeline.read_json(dir / "sheets.json")
    new = dir.parent / uuid.uuid4().hex
    try:
        new.mkdir()
        for name in (*REUSED, *(s["image"] for s in sheets)):
            if (dir / name).exists():
                shutil.copyfile(dir / name, new / name)
        _remove(new / next(s["image"] for s in sheets if s["index"] == sheet), _stair_box(stair["floors"][str(sheet)]))
        if _ink_left(new, sheets, stair, sheet) >= INK_LOW:
            raise ValueError(NOT_CLEAN)
        pipeline.write_json(new / "stage_upload.json", {"name": f"Test copy of {upload.get('name', '')}", "parent": None, "source": dir.name, "cycle": pipeline.read_json(dir / "review.json").get("project", {}).get("cycle", 1),
                                                        "note": f"Test review. The test tool removed a stair from {row.get('level') or _sheet_name(row)} on a copy of the sheet, so this is not the set as submitted."})   # no row letter, because letters are assigned again on every run
        pipeline.write_json(new / "status.json", {"state": "queued", "step": "Waiting", "stages": pipeline.stage_rows(pipeline.STAGES), "usd": 0, "sheets": len(sheets)})
    except BaseException:
        shutil.rmtree(new, ignore_errors=True)
        raise
    return new
