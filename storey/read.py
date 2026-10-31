"""Reads each sheet once: number, title, class, floor, scale, and where the main floor plan sits."""
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
from openai import BadRequestError

from .meta import spark

log = logging.getLogger(__name__)

LONG_EDGE = 2000
WORKERS = 8
CLASSES = ["floor_plan", "site_plan", "elevation", "section", "schedule", "code_summary", "cover", "detail", "other"]
TEXT_FIELDS = ("sheet_number", "sheet_title", "level", "scale")
CORNERS = ("x1", "y1", "x2", "y2")
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sheet_class", *TEXT_FIELDS, "plan_box"],
    "properties": {
        "sheet_class": {"type": "string", "enum": CLASSES},
        **{k: {"type": "string"} for k in TEXT_FIELDS},
        "plan_box": {
            "type": "object",
            "additionalProperties": False,
            "required": list(CORNERS),
            "properties": {k: {"type": "integer"} for k in CORNERS},
        },
    },
}
# The field wording comes from the read that scored 42/45 at low effort on a whole 2000 px sheet, so change it only with a new measurement.
PROMPT = """You are reading one sheet of a building drawing set. It may be in any language. Fill every field.
sheet_class: the main content of the sheet. Use code_summary for a code analysis or code data sheet. Use schedule for a sheet that is mainly tables. A plan of one floor is floor_plan, even when it also carries small schedules or is a life safety, demolition or framing plan.
sheet_number: the drawing number printed in the title block, such as A2.1, G002 or A18-1. Prefer it over a count like SHEET 161 OF 316. If the only identifier is a count, give the first number of the count. If none is printed, give an empty string.
sheet_title: the title as printed.
level: the floor this sheet shows, in the words printed in the main drawing title, such as FIRST FLOOR. Trust the main drawing title over the title block if they disagree. Empty string if the sheet does not show one floor.
scale: the scale printed next to the main drawing, copied exactly, such as 1/4" = 1'-0" or 1:150. Empty string if there is only AS SHOWN, NTS or no scale.
plan_box: the box around the walls of the main floor plan only: exclude the title block, sheet border, legend, notes, schedules, dimension strings, grid bubbles, key plans, detail insets and site work. If the sheet has no floor plan, give 0 for all four values. Coordinates are integers on a 0-1000 grid of this image, x from the left edge and y from the top edge.
"""


def _small(image):
    k = LONG_EDGE / max(image.shape[:2])
    return cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else image


def _box(grid, w, h):
    """Corners on the 0-1000 grid as sheet pixels. Corners that enclose nothing give the whole sheet, so every row has a box to crop."""
    x1, y1, x2, y2 = (min(max(v, 0), 1000) for v in grid)
    if x1 >= x2 or y1 >= y2:
        return [0, 0, w, h]
    return [x1 * w // 1000, y1 * h // 1000, x2 * w // 1000, y2 * h // 1000]


def _row(sheet, reply):
    """The stage's row for one reply. A reply is untrusted, so a wrong shape raises and the sheet counts as unread."""
    texts = [reply[k] for k in TEXT_FIELDS]
    grid = [reply["plan_box"][k] for k in CORNERS]
    cls = reply["sheet_class"]
    if cls not in CLASSES or not all(type(t) is str for t in texts) or not all(type(v) is int for v in grid):
        raise ValueError("The reply does not match the schema")
    number, title, level, scale = (t.strip() for t in texts)
    if cls == "other" and not (number or title or level or scale):
        raise ValueError("The reply is blank")   # under strict decoding, an empty answer can only arrive as blank fields
    return {"index": sheet["index"], "number": number, "title": title, "cls": cls, "level": level, "scale": scale,
            "plan_box": _box(grid, sheet["width"], sheet["height"]), "read_ok": True}


def _read(ctx, sheet):
    # BudgetReached and API errors other than a 400 pass through: the pipeline records the stage as failed, and a rerun gets the finished sheets free from the cache.
    # A 400 stays with its sheet: Meta's docs name two for one image, a content policy denial and bytes that cannot be decoded, and a resend gets the same 400.
    # shortcut: a 429, 5xx or connection error on one sheet fails the whole stage, catch it here if the API is seen failing single sheets that way.
    # shortcut: a 400 about the whole request, such as a rejected schema, marks every sheet unread and stores the stage as done, so a rerun skips it, tell it apart from a per-image 400 if one is seen.
    try:
        image = cv2.imread(str(ctx.dir / sheet["image"]))
        if image is None:
            raise ValueError("The sheet image could not be opened")
        return _row(sheet, spark(PROMPT, [_small(image)], meter=ctx.meter, schema=SCHEMA, effort="low", fresh=ctx.fresh))
    except (ValueError, KeyError, TypeError, BadRequestError) as e:
        log.warning("Sheet %d was not read: %s", sheet["index"], type(e).__name__)   # the exception text can quote the reply
        return {"index": sheet["index"], "number": "", "title": "", "cls": "other", "level": "", "scale": "",
                "plan_box": [0, 0, sheet["width"], sheet["height"]], "read_ok": False}


def read_stage(ctx) -> dict:
    """One row per sheet in sheet order. `plan_box` is in sheet pixels. A sheet with read_ok False holds no read values."""
    rows, total = [], len(ctx.sheets)
    with ThreadPoolExecutor(WORKERS) as pool:
        for row in pool.map(lambda sheet: _read(ctx, sheet), ctx.sheets):
            rows.append(row)
            ctx.progress(f"Reading sheet {len(rows)} of {total}", len(rows), total)
    return {"sheets": rows}
