"""What a set says about itself: code data, sheet index, titles, schedules, tables, door tags, room labels and exit markers. Reading only: comparing them is findings.py.

The text layer comes first and Spark reads only where no text exists. Any value a finding rests on comes from the text layer or from two agreeing Spark reads, and reads
that disagree land in `cannot_tell`. An upload is hostile, and so is the text parsed from it, so the PDF and every loop over its text run in a child process that can be killed,
as in ingest. The child reports through files, in rendered page points. Stage results are in sheet pixels.
"""
import collections
import difflib
import functools
import json
import logging
import math
import re
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from openai import BadRequestError

from .floors import NOT_MAIN, _plan_sheets
from .ingest import CPU_SECONDS, MAX_WORDS, MEMORY_BYTES
from .meta import BudgetReached, spark
from .vector import QUOTES, parse_ftin

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
COMMAND = [sys.executable, "-m", "storey.declared"]
CHILD_SECONDS = 600
WORKERS = 6
MAX_TEXT = 500           # one text value: this comes from the upload and goes to the model and on screen
LABEL_PX = 60            # half the side of the box drawn round a room label Spark pointed at
HEADING_SIZE = 12        # points: BCOE's legend prints SEE DOOR SCHEDULE at 9.6, and schedule headings on the lab's sets run 12.7 to 25
MAX_NAMES = 2000         # room names looked for on one plan, so a hostile schedule cannot make the stage walk every name on every plan
MAX_SCHEDULES = 20       # on one sheet, because each pairs with every plan and a hostile page could hold thousands
MAX_TABLES = 3           # ruled tables read on one sheet, because each costs two reads of each strip
STRIP = 0.9              # the title block sits in the right tenth of the sheet
BOILER = 0.3             # lines repeated on this share of a longer set are boilerplate
THIN_TEXT = 2000         # characters: below this a sheet's text layer is its title block, as on USC's code sheet at 303, and its tables are drawn as lines
PAIR = 40                # per mille of the sheet: Spark drifted a clinic exit marker 33 per mille between two reads
RULE_INK = 200           # grey level: at ingest size, 170 lost 14 of 95 rules on clinic sheet A5-1 and this kept them all
RULE_SHARE = 0.8         # of a table's width
MIN_RULES = 5
MIN_RULE = 0.04          # of the sheet's width, and shorter runs are drawing, not rules
RULE_ENDS = 0.01         # of the sheet's width: how far apart rules of one table may start and stop
MIN_TABLE = 0.1          # of the sheet's width, because the columns of a legend are ruled too
TITLE_EDGE = 0.96        # of the sheet's width: a run past this is the title strip or the revision table
MAX_ROW = 0.02           # of the sheet's height: the clinic index draws rules only between sections, at 0.0275, and is read in tiles instead
ROW_TOL = 3              # px, for rules of one table to count as evenly spaced
PER_STRIP = 25           # rows, the lab's 92-row sheets read at 254 of 255 in 4 strips
TB_CLIP = (0.78, 0.84, 1.0, 1.0)
WHOLE = (0.0, 0.0, 1.0, 1.0)
WIDTH_RANGE = 300, 30000   # mm: a printed figure outside this is not an exit's width
MAX_WIDTH = 20           # characters in a printed width, and int() refuses a long enough run of digits

WIDTH = re.compile(r"^([\d,]*\.?\d+)\s*(MM|M|FT|FEET|'|IN|INCH|INCHES|\")?\.?$")
WIDTH_UNITS = {"MM": 1, "M": 1000, "FT": 304.8, "FEET": 304.8, "'": 304.8, "IN": 25.4, "INCH": 25.4, "INCHES": 25.4, '"': 25.4}
SHEET_ID = r"[A-Z]{1,3}\d{0,3}[-.]?\d{1,3}(?:[.-]\d{1,3})?[A-Z]?"
SHEET_RE = re.compile(rf"^{SHEET_ID}$")
STOP = {"AND", "THE", "OF", "FOR", "A", "ONLY", "REFERENCE", "SHEET"}
FLOOR_WORDS = {"GROUND": "GROUND", "MAIN": "MAIN", "FIRST": "1", "1ST": "1", "SECOND": "2", "2ND": "2", "THIRD": "3", "3RD": "3", "3D": "3",
               "FOURTH": "4", "4TH": "4", "UPPER": "UPPER", "LOWER": "LOWER", "BASEMENT": "BASEMENT", "ATTIC": "ATTIC", "PENTHOUSE": "PENTHOUSE"}
PLAN_WORD = re.compile(r"\b(PLAN|LEVEL|FLOOR)\b", re.I)
EDITION = re.compile(r"\b((?:19|20)\d\d)\s+(?:[A-Z][A-Za-z]+\s+){0,3}(?:CBC|CRC|IBC|IRC|UBC|Code)\b|\b(?:CBC|IBC|IRC|Code)[, ]+(?:(?:19|20)\d\d)\b")
YEAR = re.compile(r"(19|20)\d\d")
NOT_USED = re.compile(r"NOT\s*USED", re.I)

SCHEDULE_KINDS = {"door": re.compile(r"DOOR\s+(AND FRAME\s+)?SCHEDULE", re.I), "window": re.compile(r"WINDOW SCHEDULE", re.I), "finish": re.compile(r"FINISH SCHEDULE", re.I)}
FIRST_COLUMN = {"MARK", "NO.", "NUMBER", "NO"}
ROOM_NO = re.compile(r"^[A-Z]?\d{3}[A-Z]?$")
LEVEL_ROW = re.compile(r"^(Ground|Main|Upper|Lower|Basement)\s+Level:?$")
TOTAL_LINE = re.compile(r"ft².*not include", re.I)
TABLE_HEADS = ((re.compile(r"OCCUPAN\w+ (CALCULATION|LOAD)", re.I), "occupant"), (re.compile(r"(EXIT|CORRIDOR) WIDTH", re.I), "exit"))
ROOM_ROW = re.compile(r"^(\d+[A-Z]?) ([A-Z].*?) ([\d,]+) SF(?: (\d+) (\d+))?$")
TOTAL_ROW = re.compile(r"^((?:Grand )?total) ([\d,]+) SF (\d+)$", re.I)
NET_NOTE = re.compile(r"\(([\d,]+) SF\) - \(.*\) = ([\d,]+) SF")
EXIT_ROW = re.compile(r"^(EXIT \S+) (\d+) (\d+) (\d+)(?: \d)? ([\d.]+) OK$")
CORRIDOR_ROW = re.compile(r"^(\d+[A-Z]? [A-Z /]+) (\d+) ([\d.]+) ([\d ]+(?:/\d+)?)\" OK$")
WORKED_ROW = re.compile(r"(\d+) OC+UPANTS \* [\d.]+\"? = ([\d.]+)\"? REQUIRED")

CODE_TITLE = re.compile(r"\bCODE\b", re.I)
CODE_WORDS = re.compile(r"\bCODES?\b|OCCUPANCY|CONSTRUCTION TYPE|TYPE OF CONSTRUCTION|SPRINKLER", re.I)
COVER_TITLE = re.compile(r"\bCOVER\b|TITLE SHEET", re.I)
INDEX_TITLE = re.compile(r"\bINDEX\b|DRAWING LIST|LIST OF DRAWINGS|SHEET LIST", re.I)
LIFE_SAFETY = re.compile(r"LIFE.?SAFETY|\bEXIT(ING)? PLAN|EGRESS PLAN", re.I)
SCHEDULE_TITLES = {"door": re.compile(r"\bDOOR\b.*\bSCHEDULE", re.I), "window": re.compile(r"\bWINDOW\b.*\bSCHEDULE", re.I), "finish": re.compile(r"\b(FINISH|ROOM)\b.*\bSCHEDULE", re.I)}
NOT_A_STOREY = re.compile(NOT_MAIN.pattern + "|furniture|finish", re.I)


def _object(**properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _array(items):
    return {"type": "array", "items": items}


STR = {"type": "string"}
INT = {"type": "integer"}

# Prompts and schemas are the ones the declared experiment scored, so change them only with a new measurement.
CODE_P = """This image is one sheet of a building drawing set. Read what it states about the building, copying values exactly as printed. Fill every field.
code_edition: the building code and its edition or year, such as International Building Code, 2012 Edition. Empty if none is printed. If a code is named with no year, give the name.
occupancy: occupancy classification or group as printed. Empty if none.
occupancy_mentions: every other place on the sheet that names an occupancy type or group, copied as printed.
construction: construction type as printed. Empty if none.
sprinklered: yes if the sheet states the building has a sprinkler system, no if it states it does not, conditional if it depends on a choice not yet made, not_stated if silent.
sprinkler_mentions: every statement on the sheet about sprinklers, copied as printed. For a table of questions with Yes and No boxes, say which box carries the mark and copy any comment.
stories: number of stories or levels as printed. Empty if none.
area: building or work area as printed. Empty if none.
occupant_load: total design occupant load as printed. Empty if none."""
CODE_S = _object(code_edition=STR, occupancy=STR, occupancy_mentions=_array(STR), construction=STR, sprinklered={"type": "string", "enum": ["yes", "no", "conditional", "not_stated"]},
                 sprinkler_mentions=_array(STR), stories=STR, area=STR, occupant_load=STR)
TB_P = """This image is one sheet of a building drawing set. Read its title block. sheet_number: the drawing number printed in the title block, such as A2.1, G002 or A18-1, not a count like SHEET 161 OF 316. sheet_title: the sheet title as printed in the title block, joining its lines. sequence: the first number of a count like SHEET 81 OF 316, empty if there is none."""
TB_S = _object(sheet_number=STR, sheet_title=STR, sequence=STR)
INDEX_P = """This image shows part of the sheet index or drawing list of a building drawing set. List every row in order, without skipping any. sheet_number: the sheet number or drawing number column. sequence: the sequence column if there is one, else empty. title: the description or title, exactly as printed. Include rows that say NOT USED. Skip empty rows and headers."""
INDEX_S = _object(rows=_array(_object(sheet_number=STR, sequence=STR, title=STR)))
STRIP_NOTE = " The image shows part of the table with its header row on top."
DOOR_P = """This image contains a door schedule table. Read the table row by row, every row, without skipping any. door: the door mark or number. width: the width column or the whole size as printed. fire_rating: the fire rating column, empty if blank or a dash. hardware: the hardware set column. room: the room name column, empty if there is no such column."""
DOOR_S = _object(rows=_array(_object(door=STR, width=STR, fire_rating=STR, hardware=STR, room=STR)))
FINISH_P = """This image contains a room finish schedule table. Read the table row by row, every row, without skipping any. room_no: the room or space number. room_name: the room name, exactly as printed."""
FINISH_S = _object(rows=_array(_object(room_no=STR, room_name=STR)))
TAG_P = """This image is part of a floor plan. Door tags are small oval or pill shaped symbols holding a door number such as 317A1, drawn beside a door. List every door tag you can read, with its text exactly as printed and x, y, the centre of the tag in the 0-1000 range of this image. Do not list room labels, keyed note circles or wall type squares."""
TAG_S = _object(tags=_array(_object(tag=STR, x=INT, y=INT)))
EXIT_P = """This image is part of a life safety plan of one floor. Point to every exit marker on the plan: a symbol the legend calls an exit location, an arrow tag at a building exit printed with an exit width and a capacity in people, or a box stating a stair well exit capacity. Do not report the legend, exit signs or fire alarm symbols. For each exit give x, y in the 0-1000 range of this image, width as printed (empty if none), and capacity, the number of people (0 if none printed). Also list every travel distance value printed on the plan, and any statement of occupant load."""
EXIT_S = _object(exits=_array(_object(x=INT, y=INT, width=STR, capacity=INT)), travel_distances=_array(STR), occupant_loads=_array(STR))
OCC_P = """This image is an occupant load table from a building code sheet. For each data row give area (the floor area as printed), factor (the floor area per occupant as printed) and occupants (the design occupant load as printed)."""
OCC_S = _object(rows=_array(_object(area=STR, factor=STR, occupants=STR)))
LABEL_P = """This image is one sheet of a building drawing set. sheet_number: the drawing number in the title block. sheet_title: the sheet title in the title block, joining its lines. drawing_title: the title of the main drawing as printed on the sheet outside the title block, empty if none."""
LABEL_S = _object(sheet_number=STR, sheet_title=STR, drawing_title=STR)
ROOM_P = """This image is part of a floor plan. Room labels are a room name with a room number in a small box under it, such as CORRIDOR over 1EC1. List every room label you can read. number: the number in the box, exactly as printed. name: the name above it, exactly as printed, with line breaks joined. x, y: the centre of the number in the 0-1000 range of this image. Skip labels cut by the image edge. Do not list device symbols such as F or FEC."""
ROOM_S = _object(rooms=_array(_object(number=STR, name=STR, x=INT, y=INT)))
JUDGE_P = """Two strings name one sheet of a building drawing set. "index" comes from the sheet index on the cover, "block" from the title block on the sheet itself. For each pair decide if they can name the same drawing. Answer same when one is an abbreviation, a longer or shorter form, a synonym, a spelling variant or a typo of the other. Answer different when they name a different floor, block, building part, trade or subject, even if most words match. Return one result per id.

"""
JUDGE_S = _object(results=_array(_object(id=STR, answer={"type": "string", "enum": ["same", "different"]})))
CODE_FIELDS = {"edition": "code_edition", "occupancy": "occupancy", "construction": "construction", "sprinklers": "sprinklered", "stories": "stories", "area": "area", "occupant_load": "occupant_load"}
CODE_MENTIONS = {"occupancy": "occupancy_mentions", "sprinklers": "sprinkler_mentions"}
SPARK_TABLES = {"door": (DOOR_P, DOOR_S, "door", {"WIDTH": "width", "FIRE RATING": "fire_rating", "HARDWARE": "hardware", "ROOM": "room"}, 12000),
                "finish": (FINISH_P, FINISH_S, "room_no", {"NAME": "room_name"}, 6000)}


def note(what, why, sheet=None):
    return {"what": what, "why": why, "sheet": sheet}


def some(items, n=10):
    items = list(items)
    return ", ".join(items[:n]) + (f" and {len(items) - n} more" if len(items) > n else "")


def norm(s):
    """Uppercase, & to AND, punctuation to space, collapsed whitespace."""
    s = (s or "").upper().replace("&", " AND ")
    return " ".join(re.sub(r"[^A-Z0-9/.]+", " ", s).replace(".", " ").split())


def tokens(s):
    """Normalised words with stop words dropped and a trailing plural S folded."""
    out = set()
    for t in norm(s).replace("/", " ").split():
        if t not in STOP:
            out.add(t[:-1] if len(t) > 3 and t.endswith("S") and not t.endswith("SS") else t)
    return out


def floor_words(s):
    return {FLOOR_WORDS[t] for t in norm(s).split() if t in FLOOR_WORDS} | set(re.findall(r"LEVEL (\d)", norm(s)))


def title_same(a, b):
    """Two titles name the same sheet: equal word sets, or one holds the other with at least two shared words, and no floor word conflict."""
    ta, tb = tokens(a), tokens(b)
    fa, fb = floor_words(a), floor_words(b)
    if fa and fb and not fa & fb:
        return False
    return ta == tb or (len(ta & tb) >= 2 and (ta <= tb or tb <= ta))


def room_same(a, b):
    """Room names match, allowing a typo, a shortened word or one name inside the other. A name on one side only is a difference."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return na == nb
    if na == nb or na in nb or nb in na or difflib.SequenceMatcher(None, na, nb).ratio() >= 0.6:
        return True
    short, long_ = sorted((na.replace("/", " ").split(), nb.replace("/", " ").split()), key=len)
    return all(any(w.startswith(s[:3]) for w in long_) for s in short)


def parse_number(text):
    """A number as printed, whole or with a fraction, or None. Upload text can hold 1.2.3, 1/0 or more digits than a float holds."""
    try:
        return float(sum(Fraction(part) for part in text.split()))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def num(s):
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(s))
    return parse_number(m[0].replace(",", "")) if m else None


def agree(a, b):
    """The value two reads share, or None. The reads must give the same words in the same order, so a word added or a digit changed is no match."""
    words = [norm(re.sub(r"(?<=\d),(?=\d)", "", v)).split() for v in (a, b)]
    return " ".join(min((a, b), key=len).split()) if words[0] and words[0] == words[1] else None


def width_mm(text):
    """A printed exit width in millimetres, or None for any form not read here. The clinic prints a bare 915, so a bare number past a door's size in inches is millimetres."""
    t = text.strip().upper().translate(QUOTES)
    if len(t) > MAX_WIDTH:
        return None
    inches, m = parse_ftin(t), WIDTH.match(t)
    if inches is not None:
        mm = inches * 25.4
    elif m:
        v = parse_number(m[1].replace(",", ""))
        mm = v * WIDTH_UNITS[m[2]] if m[2] else v * 25.4 if v < 200 else v
    else:
        return None
    return mm if WIDTH_RANGE[0] <= mm <= WIDTH_RANGE[1] else None


# ---- the child process: only this part opens the PDF ----

def save(path, obj):
    """Atomic, because the parent polls this folder, and refuses NaN and Infinity, because they stop a browser's JSON parser."""
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False))
    tmp.replace(path)


def load(path):
    """A file the child saved, or {} if it never got that far."""
    return json.loads(path.read_text()) if path.exists() else {}


def page_lines(page):
    """Text lines in the rendered frame, in page points. `dir` is the reading direction in that frame. split() also takes the non-breaking spaces CAD text carries."""
    m = page.rotation_matrix
    turn = pymupdf.Matrix(m.a, m.b, m.c, m.d, 0, 0)
    out = []
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        for line in block.get("lines", ()):
            text = " ".join(" ".join(s["text"] for s in line["spans"]).split())[:MAX_TEXT]
            if text:
                r = pymupdf.Rect(line["bbox"]) * m
                d = pymupdf.Point(line["dir"]) * turn
                out.append({"t": text, "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1, "size": round(line["spans"][0]["size"], 1), "dir": [round(d.x), round(d.y)]})
    return out[:MAX_WORDS]


def shape_tags(page):
    """Short words inside a small closed shape: [word, x, y, [w, h, kind]] in the rendered frame. A sheet with no short words skips the slow read of its strokes."""
    words = [w for w in page.get_text("words")[:MAX_WORDS] if re.fullmatch(r"[A-Za-z0-9]{1,4}", w[4].strip())]
    if not words:
        return []
    m = page.rotation_matrix
    shapes = []
    for d in page.get_drawings():
        r = d["rect"]
        if 8 <= r.width <= 32 and 8 <= r.height <= 32:
            k = collections.Counter(i[0] for i in d["items"])
            kind = "circle" if k.get("c", 0) >= 4 and abs(r.width - r.height) < 1 else "rect" if k.get("re", 0) == 1 or k.get("qu", 0) == 1 else "poly" if k.get("l", 0) >= 4 else None
            if kind:
                shapes.append((r * m, kind))
    out = []
    for w in words:
        q = pymupdf.Rect(w[:4]) * m
        cx, cy = (q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2
        hit = [s for s in shapes if abs((s[0].x0 + s[0].x1) / 2 - cx) < 4 and abs((s[0].y0 + s[0].y1) / 2 - cy) < 4 and s[0].width >= q.width - 1]
        if hit:
            s = min(hit, key=lambda s: s[0].width * s[0].height)
            out.append([w[4].strip(), cx, cy, [round(s[0].width), round(s[0].height), s[1]]])
    return out


def parse_all(job, folder):
    """The child's work. Each sheet is saved as soon as it is known, then the text reads for the set."""
    pages = {}
    with pymupdf.open(job["pdf"]) as doc:
        for i in job["sheets"]:
            try:
                page = doc[i]
                pages[i] = {"w": page.rect.width, "h": page.rect.height, "lines": page_lines(page), "tags": shape_tags(page)}
                save(folder / f"{i}.json", pages[i])
            except Exception:   # one bad page must not cost the rest, and that includes a value json refuses
                pages.pop(i, None)
                save(folder / f"{i}.json", {"failed": True})
    text = text_reads(pages)
    save(folder / "text.json", {"text": text, "picks": select(job["sheets"], {r["index"]: r for r in job["rows"]}, text, {"order": job["order"]})})


def main(argv):
    """Child process. Reports through files, because a hostile PDF can flood stdout with library warnings."""
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    except (ValueError, OSError):
        pass   # shortcut: macOS refuses this limit, the wall-clock timeout in text_pages still applies
    folder = Path(argv[0])
    parse_all(json.loads((folder / "job.json").read_text()), folder)
    return 0


def text_pages(ctx, ids, rows, floors, notes):
    """The child's pages, text reads and sheet picks, or nothing for an upload of images. Pairing schedules with plans walks the upload's text too, so the child also selects.
    A sheet the child did not finish has no page and no text read."""
    pdfs = sorted((ctx.dir / "upload").glob("*.pdf"))
    if not pdfs:
        return {}, {}, None
    folder = ctx.dir / "declared"
    shutil.rmtree(folder, ignore_errors=True)   # a rerun must not read an earlier child's files
    folder.mkdir()
    save(folder / "job.json", {"pdf": str(pdfs[0].resolve()), "sheets": ids, "rows": list(rows.values()), "order": floors.get("order")})
    child = subprocess.Popen([*COMMAND, str(folder)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + CHILD_SECONDS
    while child.poll() is None and time.monotonic() < deadline:
        ctx.progress("Reading the text layer", min(len(list(folder.glob("*.json"))) - 1, len(ids)), len(ids))
        time.sleep(0.2)
    child.kill()
    child.wait()
    done = load(folder / "text.json")
    text = {int(i): t for i, t in done.get("text", {}).items()}
    if len(text) < len(ids):
        notes.append(note("Text layer", f"The text layer of {len(ids) - len(text)} of {len(ids)} sheets could not be read, so those sheets went to Spark."))
    notes += [note("Text layer", "The tables, index and labels on this sheet could not be read from its text layer.", i) for i, t in text.items() if t.get("unread")]
    return {i: load(folder / f"{i}.json") for i in text}, text, done.get("picks")


# ---- text-layer readers: plain data in, plain data out ----

def rows_of(lines, tol=3.5):
    """Lines grouped into rows by y."""
    rows = []
    for x in sorted(lines, key=lambda a: a["y0"]):
        if rows and abs(rows[-1][0]["y0"] - x["y0"]) <= tol:
            rows[-1].append(x)
        else:
            rows.append([x])
    return [sorted(r, key=lambda a: a["x0"]) for r in rows]


def bounds(lines):
    return [min(x["x0"] for x in lines), min(x["y0"] for x in lines), max(x["x1"] for x in lines), max(x["y1"] for x in lines)]


def title_blocks(pages):
    """Sheet number and title from each page's right strip. A number blank here sends the sheet to Spark. Lines repeated across the set are boilerplate."""
    strips = {i: [x for x in p["lines"] if x["x0"] >= STRIP * p["w"]] for i, p in pages.items()}
    count = collections.Counter(t for strip in strips.values() for t in {x["t"] for x in strip})
    n = len(strips)
    limit = max(2, 0.75 * n) if n <= 8 else max(3, BOILER * n)   # short sets repeat real titles such as PLANS & AXONS
    out = {}
    for i, strip in strips.items():
        ids = [x for x in strip if SHEET_RE.match(x["t"].replace(" ", ""))]
        number = max(ids, key=lambda x: x["size"], default=None)
        rest = [x for x in strip if x is not number and count[x["t"]] < limit]
        size = number["size"] if number else max((x["size"] for x in strip), default=0)
        rest = [x for x in rest if x["size"] >= 0.3 * size and not SHEET_RE.match(x["t"].replace(" ", ""))]
        rest = [x for x in rest if not re.fullmatch(r"[A-Z]{2,3}|\d\d[/.]\d\d[/.]\d\d(\d\d)?|\d{4}\d*/\d+", x["t"])]   # drawn-by initials and dates
        turned = sum(abs(x["dir"][1]) for x in rest) > sum(abs(x["dir"][0]) for x in rest)
        rest.sort(key=(lambda x: (round(x["x0"] / 4), x["y0"])) if turned else (lambda x: (round(x["y0"] / 4), x["x0"])))
        held = rest + ([number] if number else [])
        out[i] = {"number": number["t"].replace(" ", "") if number else "", "title": " ".join(x["t"] for x in rest)[:MAX_TEXT], "box": bounds(held) if held else None}
    return out


def cover_index(lines, known):
    """Index rows: a sheet id with a title on its right. A column needs 4 rows and must mostly name sheets in the set, which keeps out stray references and lists of code sections."""
    rows = []
    for x in lines:
        m = re.match(rf"^({SHEET_ID})\s+(\S.*)$", x["t"])
        if m and x["size"] >= 8 and any(c.isalpha() for c in m[2]) and SHEET_RE.match(m[1]):
            rows.append({"number": m[1], "title": m[2], "x0": x["x0"], "y0": x["y0"], "box": bounds([x])})
        elif SHEET_RE.match(x["t"]) and x["size"] >= 8:
            near = [o for o in lines if o is not x and 0 < o["x0"] - x["x1"] < 120 and abs(o["y0"] - x["y0"]) < 3 and any(c.isalpha() for c in o["t"])]
            if near:
                o = min(near, key=lambda o: o["x0"])
                rows.append({"number": x["t"], "title": o["t"], "x0": x["x0"], "y0": x["y0"], "box": bounds([x, o])})
    columns = collections.defaultdict(list)
    for r in rows:
        columns[round(r["x0"] / 6)].append(r)
    good = [c for c in columns.values() if len(c) >= 4 and sum(r["number"] in known for r in c) >= 0.5 * len(c)]
    return [r | {"not_used": bool(NOT_USED.search(r["title"]))} for c in good for r in sorted(c, key=lambda r: r["y0"])]


def drawing_labels(lines, width):
    """View titles outside the title block: lines that name a floor and a plan, level or floor."""
    return [x for x in lines if x["x0"] < STRIP * width and PLAN_WORD.search(x["t"]) and floor_words(x["t"]) and len(x["t"]) < 60 and x["size"] >= 15]


def schedules(lines):
    """Each door, window and finish schedule as kind, title, cols, rows and box. Cells go to the nearest column centre."""
    across = [x for x in lines if x["dir"] == [1, 0]]
    out, seen = [], set()
    for kind, title in SCHEDULE_KINDS.items():
        for t in across:
            if not (title.search(t["t"]) and t["size"] >= 9 and len(t["t"]) < 60):
                continue
            near = [x for x in across if t["y0"] < x["y0"] < t["y0"] + 95 and t["x0"] - 40 < x["x0"] < t["x0"] + 720 and x is not t]
            head = next((r for r in rows_of(near) if r[0]["t"].upper() in FIRST_COLUMN and len(r) >= 3), None)
            if head is None or (kind, round(head[0]["y0"])) in seen:
                continue
            seen.add((kind, round(head[0]["y0"])))
            head = [c for i, c in enumerate(head) if i == 0 or c["x0"] - head[i - 1]["x1"] < 150]   # a stray callout on the header row is not a column
            cols = [(c["t"].upper(), (c["x0"] + c["x1"]) / 2) for c in head]
            xmin, xmax = head[0]["x0"] - 30, head[-1]["x1"] + 130
            body = [x for x in across if x["y0"] > head[0]["y0"] + 3 and xmin < x["x0"] < xmax and x["size"] <= head[0]["size"] + 1]
            rows, last = [], head[0]["y0"]
            for r in rows_of(body):
                if r[0]["y0"] - last > 45:
                    break
                row = {}
                for c in r:
                    name = min(cols, key=lambda k: abs(k[1] - (c["x0"] + c["x1"]) / 2))[0]
                    row[name] = (row.get(name, "") + " " + c["t"]).strip()
                key = cols[0][0]
                if key in row and re.fullmatch(r"[A-Z0-9.\-]{1,8}", row[key]):
                    row["_box"] = bounds(r)
                    rows.append(row)
                    last = r[0]["y0"]
                elif rows and len(r) >= 2:
                    break
            if rows:
                out.append({"kind": kind, "title": t["t"], "cols": [c[0] for c in cols], "rows": rows, "box": [min(x["x0"] for x in [*head, t]), t["y0"], max(x["x1"] for x in head), rows[-1]["_box"][3]]})
    return out


def in_boxes(x, boxes):
    return any(b[0] - 5 <= x["x0"] <= b[2] + 5 and b[1] - 5 <= x["y0"] <= b[3] + 5 for b in boxes)


def schedule_headings(lines):
    """Kinds of schedule named by a short heading line, parsed or not. A note that only mentions a schedule runs longer."""
    return sorted({kind for kind, pattern in SCHEDULE_KINDS.items() for x in lines if x["size"] >= HEADING_SIZE and (m := pattern.search(x["t"])) and len(x["t"]) - len(m[0]) <= 15})


def plan_rooms(lines, exclude, sheets=()):
    """Room numbers with the name printed above them, outside schedule boxes. A number with no name, or a sheet number, is not a room."""
    out = []
    for x in lines:
        if ROOM_NO.match(x["t"]) and not in_boxes(x, exclude):
            above = sorted((o for o in lines if 0 < x["y0"] - o["y0"] < 34 and abs((o["x0"] + o["x1"]) / 2 - (x["x0"] + x["x1"]) / 2) < 45 and re.search("[A-Za-z]{2}", o["t"]) and not in_boxes(o, exclude)), key=lambda o: -o["y0"])
            name = " ".join(o["t"] for o in above[:2][::-1])
            if name and x["t"] not in sheets:
                out.append({"no": x["t"], "name": name, "box": bounds([x, *above[:2]])})
    return out


def _band(lines, head):
    """Lines under a table heading, down to the next heading of its size."""
    # shortcut: the width comes from the one text-layer occupant table in the lab, a table wider than this reads no rows and lands in cannot_tell, widen it when a set shows one.
    column = [x for x in lines if x["dir"] == [1, 0] and head["x0"] - 60 <= x["x0"] < head["x0"] + 390 and x["y0"] > head["y0"]]
    stop = min((x["y0"] for x in column if x["size"] >= 0.9 * head["size"]), default=math.inf)
    return [x for x in column if x["y0"] < stop]


def _occupant_row(text, box):
    if m := ROOM_ROW.match(text):
        return {"room": m[1], "name": m[2], "area": num(m[3]), "factor": num(m[4]), "occupants": int(m[5]) if m[5] else None, "box": box}
    if m := TOTAL_ROW.match(text):
        return {"room": None, "name": m[1], "area": num(m[2]), "factor": None, "occupants": int(m[3]), "total": True, "box": box}
    if m := NET_NOTE.search(text):
        return {"note": text, "area": num(m[1]), "net_area": num(m[2]), "box": box}
    return None


def _exit_row(text, before, box):
    if m := EXIT_ROW.match(text):
        return {"exit": m[1], "clear_in": parse_number(m[3]), "occupants": int(m[4]), "required_in": parse_number(m[5]), "box": box}
    if m := CORRIDOR_ROW.match(text):
        return {"exit": m[1], "clear_in": parse_number(m[4]), "occupants": int(m[2]), "required_in": parse_number(m[3]), "box": box}
    if m := WORKED_ROW.search(text):
        return {"exit": before.rstrip(":"), "occupants": int(m[1]), "required_in": parse_number(m[2]), "example": True, "box": box}
    return None


def _box_of(rows):
    return [min(r["box"][0] for r in rows), min(r["box"][1] for r in rows), max(r["box"][2] for r in rows), max(r["box"][3] for r in rows)]


def text_tables(lines):
    """Occupant, exit and area tables as kind, rows and box. A heading whose rows no pattern reads comes back with empty rows, so the stage can say so."""
    out, rows = [], []
    for x in lines:
        if LEVEL_ROW.match(x["t"]):
            values = [o for o in lines if abs(o["y0"] - x["y0"]) < 3 and "ft²" in o["t"] and o["x0"] > x["x0"]]
            if values:
                rows.append({"label": x["t"].rstrip(":"), "area": num(values[0]["t"]), "box": bounds([x, values[0]])})
    total = next((x for x in lines if TOTAL_LINE.search(x["t"])), None)
    if rows and total:
        rows.append({"label": "total", "area": num(total["t"]), "total": True, "box": bounds([total])})
        out.append({"kind": "area", "rows": rows, "box": _box_of(rows)})
    for head in (x for x in lines if x["dir"] == [1, 0] and x["size"] >= 15 and len(x["t"]) < 60):   # column heads on BCOE stand on end and say Occupant Load too
        kind = next((k for pattern, k in TABLE_HEADS if pattern.search(head["t"])), None)
        if kind:
            rows, before = [], ""
            for r in rows_of(_band(lines, head)):
                text = " ".join(c["t"] for c in r)
                row = _occupant_row(text, bounds(r)) if kind == "occupant" else _exit_row(text, before, bounds(r))
                if row:
                    rows.append(row)
                before = text
            out.append({"kind": kind, "rows": rows, "box": _box_of(rows) if rows else bounds([head])})
    return out


def text_reads(pages):
    """What the text layer gives for each sheet. A page whose tables cannot be parsed keeps its title block and tags and carries `unread`."""
    blocks = title_blocks(pages)
    numbers = {b["number"] for b in blocks.values() if b["number"]}
    out = {}
    for i, p in pages.items():
        lines = p["lines"]
        out[i] = blocks[i] | {"labels": [], "index": [], "schedules": [], "headings": [], "tables": [], "rooms": [], "tags": p["tags"], "text": " ".join(x["t"] for x in lines), "chars": sum(len(x["t"]) for x in lines)}
        try:
            found = schedules(lines)[:MAX_SCHEDULES]
            out[i] |= {"labels": [x["t"] for x in drawing_labels(lines, p["w"])], "index": cover_index(lines, numbers), "schedules": found, "headings": schedule_headings(lines), "tables": text_tables(lines),
                       "rooms": plan_rooms(lines, [t["box"] for t in found], numbers)}
        except Exception:   # one odd line must not cost the set its text layer
            out[i]["unread"] = True
    return out


def _family(tags, table):
    """The plan's tags that belong to this schedule. Door tags are the digit family that best covers its marks, a window tag sits in a polygon or a box, and each drawing style draws one shape for one kind."""
    marks = {r[table["cols"][0]] for r in table["rows"]}
    members = collections.defaultdict(list)
    for t in tags:
        members[tuple(t[3])].append(t)
    if table["kind"] == "window":
        ranked = sorted((f for f in members if f[2] in ("poly", "rect")), key=lambda f: -len(members[f]))
        fam = next((f for f in ranked if {t[0] for t in members[f]} & marks), None)
        return [t for t in tags if fam and t[3][2] == fam[2] and abs(t[3][0] - fam[0]) <= 1 and abs(t[3][1] - fam[1]) <= 1]
    ranked = sorted(members, key=lambda f: -sum(t[0] in marks for t in members[f]))
    fam = next((f for f in ranked if f[2] in ("circle", "rect") and all(t[0].isdigit() for t in members[f])), None)
    return [t for t in tags if fam and t[3][:2] in ([fam[0], fam[1]], [fam[0], fam[1] - 1]) and t[3][2] == fam[2]]


def _overlap(tags, table):
    return {t[0] for t in _family(tags, table)} & {r[table["cols"][0]] for r in table["rows"]}


# ---- which sheets each read uses ----

def select(ids, rows, text, floors):
    """Which sheets each read uses, with no model call, so the choice can be measured on sets the product has not read. `rows` and `text` are by sheet index, and either can miss a sheet."""
    def title(i):
        return " ".join(filter(None, (rows.get(i, {}).get("title"), text.get(i, {}).get("title"))))

    def cls(i):
        return rows.get(i, {}).get("cls")

    def chars(i):
        return text.get(i, {}).get("chars", 0)

    def tables(i):
        return text.get(i, {}).get("schedules", ())

    def letter(i):
        return (text.get(i, {}).get("number") or rows.get(i, {}).get("number") or "")[:1]

    def series(i):
        """Plan sheets with sheet i's letter. A demolition, ceiling or furniture plan is not where a schedule's tags live."""
        return [j for j in ids if j != i and letter(i) and letter(j) == letter(i) and "PLAN" in title(j).upper() and not NOT_A_STOREY.search(title(j))]

    life = [i for i in ids if LIFE_SAFETY.search(title(i))]
    code = [i for i in ids if cls(i) == "code_summary" or CODE_TITLE.search(title(i))]
    covers = [i for i in ids if cls(i) == "cover" or COVER_TITLE.search(title(i))][:1]
    if not code:   # the lab's small sets print code data on the cover, and its clinic in the notes of the life safety sheets
        cover = [i for i in covers if not chars(i) or CODE_WORDS.search(text[i]["text"])]
        code = sorted({*cover, *life})
    index = {"text": [], "spark": [], "cover": []}
    seen = set()
    for i in sorted((i for i in ids if text.get(i, {}).get("index")), key=lambda i: -len(text[i]["index"])):
        numbers = {r["number"] for r in text[i]["index"]}
        if not numbers & seen:   # one set repeats part of its index on the structural notes sheet
            index["text"].append(i)
            seen |= numbers
    index["text"].sort()
    if not index["text"]:
        index["spark"] = [i for i in ids if INDEX_TITLE.search(title(i))]
        index["cover"] = [] if index["spark"] else covers
    spark_tables = [[i, kind] for i in ids for kind, pattern in SCHEDULE_TITLES.items()
                    if (pattern.search(title(i)) or kind in text.get(i, {}).get("headings", ())) and kind not in {t["kind"] for t in tables(i)}]
    named = {i for i, _ in spark_tables}
    # shortcut: a schedule drawn on a plan sheet with no text layer is not found, because nothing in its title names it, ask Spark about each plan if a real set shows this.
    unplaced = [i for i in ids if i not in named and ((not chars(i) and not rows.get(i, {}).get("read_ok")) or (chars(i) < THIN_TEXT and cls(i) == "schedule" and not tables(i)))]
    from_rows = [r["index"] for r in _plan_sheets(list(rows.values())) if not NOT_A_STOREY.search(r["title"])]
    storeys = [i for i in floors.get("order") or from_rows if i in ids]
    tags, found_tags, rooms, unread = [], {}, [], set()
    no_tags = any(kind == "door" for _, kind in spark_tables)
    no_rooms = any(kind == "finish" for _, kind in spark_tables)

    def readable(plans):
        """The plans with a text layer. A plan with none goes on `unread`, because the schedule's comparison misses what it holds."""
        unread.update(j for j in plans if j not in text)
        return [j for j in plans if j in text]

    for i in ids:
        own = "PLAN" in title(i).upper()
        for n, table in enumerate(tables(i)):
            if table["kind"] == "finish":
                found = [i] if own else readable(series(i))
                rooms += found
                no_rooms = no_rooms or not found
                continue
            # the lab paired each schedule with its one best plan, because its sets hold one plan a schedule. A commercial set serves every storey from one sheet.
            plans = [i] if own and _overlap(text[i]["tags"], table) else [j for j in readable(series(i)) if _overlap(text[j]["tags"], table)]
            if plans:
                tags.append([i, n, plans])
                found_tags |= {(p, table["kind"], word, x, y): True for p in plans for word, x, y, _ in _family(text[p]["tags"], table)}
            elif table["kind"] == "door":
                no_tags = True
    if no_rooms:   # USC draws its finish schedule as lines and keeps its plan's room numbers in the text layer
        rooms += [p for p in storeys if text.get(p, {}).get("rooms")]
    return {"code": code, "index": index, "life_safety": life, "schedules": spark_tables, "tags": {"text": tags, "spark": storeys if no_tags else [], "found": [list(t) for t in found_tags]},
            "rooms": {"text": sorted(set(rooms)), "spark": [p for p in storeys if p not in rooms] if no_rooms else []},
            "occupant": [i for i in code if chars(i) < THIN_TEXT and not text.get(i, {}).get("tables")], "unplaced": unplaced, "unread": sorted(unread)}


# ---- crops for Spark ----

def tiles(cols, rows, overlap=0.08, box=WHOLE):
    """Clips of the box in 0-1 sheet coordinates, each overlapping its neighbours."""
    x0, y0, x1, y1 = box
    w, h = (1 + overlap) / cols, (1 + overlap) / rows
    out = []
    for i in range(cols):
        for j in range(rows):
            left, top = i * (1 - w) / max(cols - 1, 1), j * (1 - h) / max(rows - 1, 1)
            out.append((x0 + left * (x1 - x0), y0 + top * (y1 - y0), x0 + (left + w) * (x1 - x0), y0 + (top + h) * (y1 - y0)))
    return out


def crop(image, clip, px):
    """A 0-1 clip with long side px, up as well as down, as the lab rendered."""
    # shortcut: crops come from the ingest image, about 2 px per point on a 42 in sheet where the lab rendered plan labels at 4.3, so labels are enlarged but gain no detail, render tiles in the child if label reads fall short.
    if image is None:
        raise ValueError("The sheet image could not be opened")
    h, w = image.shape[:2]
    part = image[int(clip[1] * h):int(clip[3] * h), int(clip[0] * w):int(clip[2] * w)]
    k = px / max(part.shape[:2])
    return cv2.resize(part, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)


def table_boxes(image):
    """Ruled tables as [x0, y0, x1, y1] in pixels, most rules first, then widest and tallest. A table is a group of long rules that share both ends."""
    # shortcut: a skewed scan breaks its rules into short runs and finds no table, deskew if real scans are worth reading.
    h, w = image.shape[:2]
    dark = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < RULE_INK).astype(np.uint8)
    runs = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, max(20, int(MIN_RULE * w))), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
    groups = []
    for left, top, width, *_ in sorted(stats[1:, :4].tolist()):
        for g in groups:
            if abs(g[0][0] - left) <= RULE_ENDS * w and abs(g[0][2] - (left + width)) <= RULE_ENDS * w:
                g.append((left, top, left + width))
                break
        else:
            groups.append([(left, top, left + width)])
    found = []
    for g in groups:
        ys = sorted(top for _, top, _ in g)
        x0, x1 = min(a for a, _, _ in g), max(b for _, _, b in g)
        if len(g) < MIN_RULES or x1 - x0 < MIN_TABLE * w or x1 > TITLE_EDGE * w or np.median(np.diff(ys)) > MAX_ROW * h:
            continue
        margin = max(2, (x1 - x0) // 100)   # inside the table, so a sheet border beside it cannot look like a tall header
        inked = dark[:, x0 + margin:x1 - margin].any(1)
        above, below = np.flatnonzero(~inked[:ys[0]]), np.flatnonzero(~inked[ys[-1]:])
        y0 = int(above[-1]) + 1 if len(above) else 0     # a tall header sits above the first rule that reaches both ends
        y1 = ys[-1] + int(below[0]) if len(below) else h   # and a last row closed by the sheet border sits below the last one
        found.append((len(g), (x1 - x0) * (ys[-1] - y0), [int(x0), y0, int(x1), y1]))
    return [box for _, _, box in sorted(found, key=lambda f: f[:2], reverse=True)]


def rule_rows(image):
    """y of the rules that cross most of a table."""
    ink = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < RULE_INK).mean(1)
    rows = np.flatnonzero(ink > RULE_SHARE)
    return [int(g.mean()) for g in np.split(rows, np.flatnonzero(np.diff(rows) > 2) + 1) if len(g)]


def row_strips(image, box, per_strip=PER_STRIP, px=2000):
    """([(strip image, its rows' box in the sheet), ...], rows found). Each strip is cut on the table's own rules and holds each row once, under a header that stops at its bottom rule.
    A table with no run of even rows goes whole, with 0 rows found."""
    x0, y0, x1, y1 = box
    table = image[y0:y1, x0:x1]
    ys = rule_rows(table)
    gaps = np.diff(ys)
    height = float(np.median(gaps)) if len(gaps) else 0.0
    start = next((i for i in range(len(gaps) - 4) if all(abs(g - height) <= ROW_TOL for g in gaps[i:i + 5])), None)
    if start is None:
        return [(_shrunk(table, px), list(box))], 0
    body = ys[start:]
    if table.shape[0] - body[-1] > height / 2:
        body = [*body, table.shape[0] - 3]   # rows continue below the last rule, so the table's bottom edge closes the last one
    rows = len(body) - 1
    head = table[max(0, ys[0] - 4):body[0] + 3]
    cuts = np.linspace(0, rows, math.ceil(rows / per_strip) + 1, dtype=int)
    return [(_shrunk(np.vstack([head, table[body[a] + 2:body[b] + 3]]), px), [x0, y0 + body[a], x1, y0 + body[b]]) for a, b in zip(cuts, cuts[1:])], rows


def _shrunk(image, px):
    k = px / max(image.shape[:2])
    return cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else image


# ---- merging reads: plain data in, plain data out ----

def field(reply, key):
    """A string field of a reply, stripped. A reply is untrusted, so anything else counts as blank."""
    value = reply.get(key) if isinstance(reply, dict) else None
    return value.strip() if isinstance(value, str) else ""


def rows_in(reply, name, keys):
    """A reply's rows, or None if the reply is lost or has the wrong shape."""
    rows = reply.get(name) if isinstance(reply, dict) else None
    if not isinstance(rows, list) or not all(isinstance(r, dict) and all(isinstance(r.get(k), str) for k in keys) for r in rows):
        return None
    return rows


def lost_parts(reads, name, keys):
    """How many parts have a reply lost or of the wrong shape in either read. `reads` holds two lists of replies, one reply per part."""
    return sum(rows_in(a, name, keys) is None or rows_in(b, name, keys) is None for a, b in zip(*reads))


def part_note(what, lost, parts, sheet):
    return note(what, f"{lost} of {parts} crops of this sheet could not be read, so entries can be missing.", sheet)


def mentions(replies, key):
    """The statements both reads make, in the first read's order. A reply is untrusted, so anything but a list of strings counts as none."""
    lists = [r.get(key) if isinstance(r, dict) else None for r in replies]
    first, second = [[m for m in x if isinstance(m, str)] if isinstance(x, list) else [] for x in lists]
    return [m for m in first if any(agree(m, other) for other in second)]


def code_fields(reads):
    """`reads` maps each code sheet, in the order tried, to its two replies. Each field takes the first sheet whose two reads agree on a value.
    `reads` in a row counts the reads behind it: 2 with no value means both reads found nothing printed, and under 2 with no value means Cannot tell. A value with `reads` 0 is an edition code_data took from the text layer.
    Occupancy and sprinklers also carry `mentions`, the other statements that sheet makes on them, for the check that a code sheet is consistent."""
    out, notes = {}, []
    for name, key in CODE_FIELDS.items():
        best = None
        for sheet, replies in reads.items():
            lost = [r for r in replies if not isinstance(r, dict) or not isinstance(r.get(key), str)]
            values = [field(r, key).replace("not_stated", "") for r in replies if r not in lost]
            if len(values) < 2:
                row, why = {"value": None, "sheet": sheet, "box": None, "reads": len(values)}, "A read of this sheet was lost."
            elif values[0] == values[1] == "":
                row, why = {"value": None, "sheet": sheet, "box": None, "reads": 2}, None
            else:
                value = agree(*values)
                row, why = {"value": value, "sheet": sheet, "box": None, "reads": 2 if value else 0}, None if value else f"Two reads gave {values[0]} and {values[1]}."
            if row["value"]:
                best, whys = row, None
                break
            if best is None or (best["reads"] == 2 and row["reads"] < 2):
                best, whys = row, why
        if best:
            out[name] = best
            if name in CODE_MENTIONS:
                best["mentions"] = mentions(reads[best["sheet"]], CODE_MENTIONS[name])
            if whys:
                notes.append(note(f"Code field {name}", whys, best["sheet"]))
    return out, notes


def _exit_rows(reply):
    rows = reply.get("exits") if isinstance(reply, dict) else None
    if not isinstance(rows, list) or not all(isinstance(r, dict) and type(r.get("x")) is int and type(r.get("y")) is int and isinstance(r.get("width"), str) and type(r.get("capacity")) is int for r in rows):
        return None
    return rows


def exit_sheet(sheet, size, clips, replies):
    """One life safety sheet from two reads of its tiles: (row for the stage, cannot_tell rows). A marker counts only from the tile whose half of the sheet holds it, so an overlap cannot double it.
    Pull stations and detectors read as exits, so a marker needs a capacity or a width that reads as one. The sum needs every tile of both reads and two equal totals."""
    reads, whole = [], True
    for read in replies:
        marks = []
        for clip, reply in zip(clips, read):
            rows = _exit_rows(reply)
            if rows is None:
                whole = False
                continue
            for r in rows:
                if not (0 <= r["x"] <= 1000 and 0 <= r["y"] <= 1000):   # first, because a reply's integer can be too long to divide
                    continue
                x, y = clip[0] + r["x"] / 1000 * (clip[2] - clip[0]), clip[1] + r["y"] / 1000 * (clip[3] - clip[1])
                if (x < 0.5) == (clip[0] < 0.25) and (y < 0.5) == (clip[1] < 0.25) and (r["capacity"] > 0 or width_mm(r["width"]) is not None):
                    marks.append({"x": x, "y": y, "width": r["width"], "capacity": max(r["capacity"], 0)})
        reads.append(marks)
    first, second = reads
    a, b = sum(m["capacity"] for m in first), sum(m["capacity"] for m in second)
    free = list(second)
    markers = []
    for m in first:
        twin = next((o for o in free if o["capacity"] == m["capacity"] and width_mm(o["width"]) == width_mm(m["width"]) and 1000 * math.hypot(o["x"] - m["x"], o["y"] - m["y"]) <= PAIR), None)
        if twin:
            free.remove(twin)
            markers.append({"point": [round(m["x"] * size[0]), round(m["y"] * size[1])], "width_mm": width_mm(m["width"]), "capacity": m["capacity"] or None})
    row = {"sheet": sheet, "sum": a if whole and a == b and a else None, "reads": [a, b] if whole else None, "markers": markers}
    if not whole:
        why = "A tile of this sheet came back empty, so its exits are not summed."
    elif a != b:
        why = f"Two reads summed to {a} and {b} people."
    elif not a:
        why = "No exit capacity was read on this sheet."
    else:
        return row, []
    return row, [note("Exit capacity", why, sheet)]


def index_rows(sheet, reads, notes):
    """Rows of one index sheet that both reads agree on. `reads` holds two lists of replies, one per clip."""
    runs = [{}, {}]
    for run, replies in zip(runs, reads):
        for reply in replies:
            for r in rows_in(reply, "rows", ("sheet_number", "title")) or ():
                number = field(r, "sheet_number")
                if number not in ("", "-"):
                    run.setdefault(number, field(r, "title"))
    both = {n: t for n, t in runs[0].items() if n in runs[1] and title_same(t, runs[1][n])}
    left = [n for n in {**runs[0], **runs[1]} if n not in both]
    if left:
        notes.append(note("Sheet index", f"Two reads did not agree on the rows for {some(left)}.", sheet))
    if lost := lost_parts(reads, "rows", ("sheet_number", "title")):
        notes.append(part_note("Sheet index", lost, len(reads[0]), sheet))
    return [{"number": n, "title": t, "not_used": bool(NOT_USED.search(t)), "sheet": sheet, "box": None} for n, t in both.items()]


def spark_schedule(sheet, kind, boxes, reads, notes):
    """One schedule from two reads of its strips. A row needs both reads, and a cell the two reads differ on is left empty. A ruled table has rows, so a strip that comes back with none counts as lost."""
    _, _, key, cells, _ = SPARK_TABLES[kind]
    keys = (key, *cells.values())
    if lost := sum(not (rows_in(a, "rows", keys) and rows_in(b, "rows", keys)) for a, b in zip(*reads)):
        notes.append(part_note(f"{kind.capitalize()} schedule", lost, len(boxes), sheet))
    runs = [{}, {}]
    for run, replies in zip(runs, reads):
        for box, reply in zip(boxes, replies):
            for r in rows_in(reply, "rows", keys) or ():
                mark = norm(r[key]).replace(" ", "")
                if mark:
                    run.setdefault(mark, (r, box))
    rows, differ = [], []
    for mark, (r, box) in runs[0].items():
        if mark in runs[1]:
            other = runs[1][mark][0]
            same = {c: room_same(field(r, k), field(other, k)) if c in ("NAME", "ROOM") else norm(field(r, k)) == norm(field(other, k)) for c, k in cells.items()}
            rows.append({"mark": mark, "cells": {c: field(r, k) if same[c] else None for c, k in cells.items()}, "box": box})
            if not all(same.values()):
                differ.append(mark)
    once = [m for m in {**runs[0], **runs[1]} if not (m in runs[0] and m in runs[1])]
    if differ or once:
        notes.append(note(f"{kind.capitalize()} schedule", " ".join(filter(None, [f"Only one read saw {some(once)}." if once else "", f"Two reads differ on a cell of {some(differ)}." if differ else ""])), sheet))
    return {"sheet": sheet, "kind": kind, "rows": rows, "source": "spark"} if rows else None


def schedule_score(table):
    """Rows that fill the kind's first cell, then rows. A door schedule leads with a width, so a hardware table read as doors scores low."""
    rows = table["rows"] if table else ()
    return sum(bool(next(iter(r["cells"].values()))) for r in rows), len(rows)


def tile_points(clips, reads, name, key, size):
    """{mark: (text, point)} per read from tiled plan reads, keeping a mark's first sighting. Marks are normalised, so `1e 08` is `1E08`."""
    out = []
    for replies in reads:
        run = {}
        for clip, reply in zip(clips, replies):
            for r in rows_in(reply, name, (key,)) or ():
                x, y = r.get("x"), r.get("y")
                mark = norm(r[key]).replace(" ", "").replace("-", "")
                if mark and type(x) is int and type(y) is int and 0 <= x <= 1000 and 0 <= y <= 1000:
                    run.setdefault(mark, (r, [round((clip[0] + x / 1000 * (clip[2] - clip[0])) * size[0]), round((clip[1] + y / 1000 * (clip[3] - clip[1])) * size[1])]))
        out.append(run)
    return out


def plan_gaps(what, reads, name, key, found, sheet):
    """Cannot tell rows for a plan read in tiles: the tiles lost, or a plan that gave nothing both reads agree on."""
    if lost := lost_parts(reads, name, (key,)):
        return [part_note(what, lost, len(reads[0]), sheet)]
    return [] if found else [note(what, "No entry on this plan was given by both reads.", sheet)]


def not_compared(tables, rooms):
    """Cannot tell rows for schedules that nothing on a plan can be compared with. A finish schedule meets its plans through `rooms`, and the other kinds through `plans`."""
    out = [note(f"{t['kind'].capitalize()} schedule", "No plan's tags answer to this schedule, so its marks were not compared with a plan.", t["sheet"]) for t in tables if t["kind"] != "finish" and not t["plans"]]
    if not rooms and any(t["kind"] == "finish" for t in tables):
        out.append(note("Room labels", "No room label was read on a plan, so the finish schedule's names were not compared with one."))
    return out


# ---- the stage ----

class Reads:
    """Runs Spark jobs in parallel. A lost read gives None. A spent budget stops the reads left and is kept for one note. Other API errors pass through, as in read.py, so the stage shows as failed."""

    def __init__(self, ctx):
        self.ctx, self.stopped = ctx, None

    def one(self, job):
        key, run, prompt, image, schema, effort, tokens = job
        if self.stopped:
            return None
        try:
            return spark(prompt, [image()] if image else [], meter=self.ctx.meter, schema=schema, effort=effort, max_tokens=tokens, fresh=self.ctx.fresh, run=run)
        except BudgetReached as e:
            self.stopped = str(e)
        except (ValueError, KeyError, TypeError, ZeroDivisionError, BadRequestError, cv2.error) as e:
            log.warning("A read of %s was lost: %s", key[0], type(e).__name__)   # the exception text can quote the reply
        return None

    def __call__(self, jobs, step):
        out = {}
        with ThreadPoolExecutor(WORKERS) as pool:
            for job, reply in zip(jobs, pool.map(self.one, jobs)):
                out[job[0], job[1]] = reply
                self.ctx.progress(step, len(out), len(jobs))
        return out


def twice(key, prompt, image, schema, effort, tokens):
    return [(key, run, prompt, image, schema, effort, tokens) for run in (0, 1)]


def share(box, size):
    """A pixel box as a 0-1 clip."""
    return box[0] / size[0], box[1] / size[1], box[2] / size[0], box[3] / size[1]


def both(got, tag, i, parts):
    """Two lists of replies, one per read, each with one reply per part."""
    return [[got[(tag, i, n), run] for n in range(parts)] for run in (0, 1)]


@dataclass
class Sheets:
    """One review's sheets in the forms the reads need."""
    ids: list
    rows: dict
    sizes: dict
    pages: dict
    text: dict
    picks: dict
    load: Callable
    notes: list

    def scale(self, box, i):
        k = self.sizes[i][0] / self.pages[i]["w"]
        return [round(v * k) for v in box]

    def clipped(self, i, clip, px=2000):
        return lambda: crop(self.load(i), clip, px)

    def tiled(self, tag, i, clips, prompt, schema, effort, tokens):
        return [job for n, clip in enumerate(clips) for job in twice((tag, i, n), prompt, self.clipped(i, clip), schema, effort, tokens)]

    def tables_on(self, i):
        image = self.load(i)
        return table_boxes(image) if image is not None else []

    def plan_tiles(self, i, cols, rows):
        box = self.rows.get(i, {}).get("plan_box")
        return tiles(cols, rows, box=share(box, self.sizes[i]) if box else WHOLE)


def first_reads(s):
    """Every Spark job that needs no other read's answer, and the crops behind them."""
    picks, notes = s.picks, s.notes
    jobs = [(("title", i), 0, TB_P, s.clipped(i, TB_CLIP, 1000), TB_S, "low", 2000) for i in s.ids if not s.text.get(i, {}).get("number")]
    jobs += [job for i in picks["code"] for job in twice(("code", i), CODE_P, s.clipped(i, WHOLE), CODE_S, "low", 6000)]
    index = {i: [WHOLE] for i in picks["index"]["cover"]} | {i: [share(b, s.sizes[i]) for b in s.tables_on(i)] or tiles(4, 2) for i in picks["index"]["spark"]}
    for i, clips in index.items():
        jobs += s.tiled("index", i, clips, INDEX_P, INDEX_S, "medium", 8000)
    exits = {i: tiles(2, 2) for i in picks["life_safety"]}
    for i, clips in exits.items():
        jobs += s.tiled("exits", i, clips, EXIT_P, EXIT_S, "medium", 6000)
    strips = {}
    for i, kind in picks["schedules"]:
        if kind not in SPARK_TABLES:
            notes.append(note("Window schedule", "A window schedule with no text layer is not read.", i))
            continue
        boxes = s.tables_on(i)
        if not boxes:
            notes.append(note(f"{kind.capitalize()} schedule", "No ruled table was found on this sheet, so it was not read.", i))
            continue
        if len(boxes) > 1:
            # shortcut: a schedule split across two tables keeps only its larger part, merge tables that fill the same columns if a real set shows this.
            notes.append(note(f"{kind.capitalize()} schedule", f"This sheet holds {len(boxes)} ruled tables. The largest {min(len(boxes), MAX_TABLES)} were read and only the one with the most agreed rows counts, so entries can be missing.", i))
        prompt, schema, _, _, tokens = SPARK_TABLES[kind]
        strips[i, kind] = [row_strips(s.load(i), box)[0] for box in boxes[:MAX_TABLES]]
        for t, cut in enumerate(strips[i, kind]):
            for n, (strip, _) in enumerate(cut):
                jobs += twice(("schedule", i, kind, t, n), prompt + STRIP_NOTE, lambda strip=strip: strip, schema, "medium", tokens)
    tags = {i: s.plan_tiles(i, 2, 2) for i in picks["tags"]["spark"]}
    rooms = {i: s.plan_tiles(i, 3, 3) for i in picks["rooms"]["spark"]}
    for i, clips in tags.items():
        jobs += s.tiled("tags", i, clips, TAG_P, TAG_S, "medium", 6000)
    for i, clips in rooms.items():
        jobs += s.tiled("rooms", i, clips, ROOM_P, ROOM_S, "medium", 8000)
    occupant = {i: s.tables_on(i)[:MAX_TABLES] for i in picks["occupant"]}
    for i, boxes in occupant.items():
        for n, box in enumerate(boxes):
            jobs += twice(("occupant", i, n), OCC_P, s.clipped(i, share(box, s.sizes[i]), 1600), OCC_S, "low", 3000)
    return jobs, {"index": index, "exits": exits, "strips": strips, "tags": tags, "rooms": rooms, "occupant": occupant}


def sheet_index(s, crops, got):
    index = [{"number": r["number"], "title": r["title"], "not_used": r["not_used"], "sheet": i, "box": s.scale(r["box"], i)} for i in s.picks["index"]["text"] for r in s.text[i]["index"]]
    for i, clips in crops["index"].items():
        index += index_rows(i, both(got, "index", i, len(clips)), s.notes)
    if not index:
        s.notes.append(note("Sheet index", "No sheet index was read in this set."))
    return index


def sheet_key(number):
    return norm(number).replace(" ", "")


def sheet_titles(s, index, got, reads):
    """One row per sheet. A Spark title gets a second read only where the first disagrees with the index, and a label read only where its floor words conflict with the title.
    The title judge is a Spark call, so it runs here and evaluate stays free of model calls."""
    on_index = {}
    for r in index:
        on_index.setdefault(sheet_key(r["number"]), []).append(r)   # a number listed twice is the index check's business, and the sheet may match either row
    titles = {}
    for i in s.ids:
        block, row = s.text.get(i, {}), s.rows.get(i, {})
        labels = "; ".join(block.get("labels", ()))[:MAX_TEXT] or None
        if block.get("number"):
            titles[i] = {"sheet": i, "number": block["number"], "title": block["title"], "under_drawing": labels, "source": "text", "judge": None, "box": s.scale(block["box"], i)}
        else:   # a crop can miss a block drawn elsewhere, and the sheet read looked at the whole sheet
            reply = got[("title", i), 0]
            titles[i] = {"sheet": i, "number": field(reply, "sheet_number") or row.get("number", ""), "title": field(reply, "sheet_title") or block.get("title") or row.get("title", ""),
                         "under_drawing": labels or row.get("level") or None, "source": "spark", "judge": None, "box": [round(TB_CLIP[n] * s.sizes[i][n % 2]) for n in range(4)]}

    def matches(number, title):
        return any(title_same(r["title"], title) for r in on_index.get(sheet_key(number), ()))

    by_spark = [t for t in titles.values() if t["source"] == "spark"]
    for t in by_spark:   # the corner crop held the number on 161 of 161 text sheets and the title on 69, and a title sharing no word with the sheet read is the crop missing it, not a defect
        whole_sheet = s.rows.get(t["sheet"], {}).get("title", "")
        if not matches(t["number"], t["title"]) and matches(t["number"], whole_sheet) and not tokens(t["title"]) & tokens(whole_sheet):
            t["title"] = whole_sheet
    second = [t["sheet"] for t in by_spark if on_index and t["number"] and not matches(t["number"], t["title"])]
    conflicts = [t["sheet"] for t in by_spark if floor_words(t["title"]) and floor_words(t["under_drawing"]) and not floor_words(t["title"]) & floor_words(t["under_drawing"])]
    jobs = [(("title", i), 1, TB_P, s.clipped(i, TB_CLIP, 1000), TB_S, "low", 2000) for i in second]
    jobs += [job for i in conflicts for job in twice(("label", i), LABEL_P, s.clipped(i, WHOLE), LABEL_S, "low", 3000)]
    again = reads(jobs, "Checking titles against the index")
    unsure = set()
    for i in second:
        t, reply = titles[i], again[("title", i), 1]
        number = field(reply, "sheet_number")
        if sheet_key(number) != sheet_key(t["number"]):   # the sheet stays unread, so no finding can rest on either number
            s.notes.append(note("Sheet number", f"Two reads gave {t['number']} and {number or 'nothing'}.", i))
            t |= {"number": "", "title": ""}
            unsure.add(i)
        elif not title_same(t["title"], field(reply, "sheet_title")):
            s.notes.append(note("Sheet title", f"Two reads gave {t['title']} and {field(reply, 'sheet_title') or 'nothing'}.", i))
            unsure.add(i)
    for i in conflicts:
        replies = again[("label", i), 0], again[("label", i), 1]
        label = agree(field(replies[0], "drawing_title"), field(replies[1], "drawing_title"))
        titles[i]["under_drawing"] = label
        if label is None:
            s.notes.append(note("Label under the drawing", "Two reads did not agree on it.", i))
        elif any(floor_words(field(r, "sheet_title")) != floor_words(titles[i]["title"]) for r in replies):   # the crop is one read, and a floor conflict needs two behind it
            s.notes.append(note("Sheet title", f"The title read gave {titles[i]['title']}, and the reads of the whole sheet did not both give its floor.", i))
            titles[i]["under_drawing"] = None
            unsure.add(i)
    for t in titles.values():
        if not (t["number"] and t["title"]) and t["sheet"] not in unsure:
            s.notes.append(note("Sheet title", "The number or title of this sheet could not be read.", t["sheet"]))
    pairs = [{"id": str(t["sheet"]), "index": on_index[sheet_key(t["number"])][0]["title"], "block": t["title"]} for t in titles.values()
             if t["title"] and t["sheet"] not in unsure and sheet_key(t["number"]) in on_index and not matches(t["number"], t["title"])]
    if pairs:
        judged = reads(twice(("judge",), JUDGE_P + json.dumps(pairs, indent=0), None, JUDGE_S, "low", 6000), "Comparing titles")
        answers = [{r["id"]: r["answer"] for r in rows_in(judged[("judge",), run], "results", ("id", "answer")) or ()} for run in (0, 1)]
        for p in pairs:
            t, verdicts = titles[int(p["id"])], [a.get(p["id"]) for a in answers]
            if None in verdicts:
                s.notes.append(note("Sheet title", f"The title check gave no answer for {t['number']}.", t["sheet"]))
            else:
                t["judge"] = "different" if verdicts == ["different", "different"] else "same"
    return [titles[i] for i in s.ids]


def code_data(s, got):
    code, notes = code_fields({i: [got[("code", i), 0], got[("code", i), 1]] for i in s.picks["code"]})
    s.notes += notes
    if not s.picks["code"]:
        s.notes.append(note("Code summary", "No code sheet was found in this set."))
    if code and not YEAR.search(code["edition"]["value"] or ""):
        found = next(((i, m[0]) for i in dict.fromkeys([*s.picks["code"], *s.ids[:3]]) if (m := EDITION.search(s.text.get(i, {}).get("text", "")))), None)
        if found:
            code["edition"] |= {"sheet": found[0], "value": found[1], "reads": 0}
            s.notes[:] = [n for n in s.notes if n["what"] != "Code field edition"]
    for row in code.values():
        wanted = norm(row["value"])
        line = next((x for x in s.pages.get(row["sheet"], {}).get("lines", ()) if wanted and wanted in norm(x["t"])), None)
        if line:
            row["box"] = s.scale(bounds([line]), row["sheet"])
    return code


def text_schedules(s):
    """Schedules on the text layer, and the plan tags that answer to them."""
    paired = {(i, n): plans for i, n, plans in s.picks["tags"]["text"]}
    out = []
    for i in s.ids:
        for n, t in enumerate(s.text.get(i, {}).get("schedules", ())):
            key = t["cols"][0]
            rows = [{"mark": r[key], "cells": {c: r[c] for c in t["cols"][1:] if c in r}, "box": s.scale(r["_box"], i)} for r in t["rows"]]
            out.append({"sheet": i, "kind": t["kind"], "rows": rows, "source": "text", "plans": paired.get((i, n)) or (s.picks["tags"]["spark"] if t["kind"] == "door" else [])})
    return out, [{"sheet": p, "kind": kind, "mark": word, "point": s.scale([x, y], p)} for p, kind, word, x, y in s.picks["tags"]["found"]]


def spark_schedules(s, crops, got):
    """One schedule per sheet and kind. The best score among the tables read wins, and a tie goes to the largest. Two kinds that keep one table cannot both be right, so neither is kept."""
    kept = {}
    for (i, kind), tables in crops["strips"].items():
        results, unread = [], []
        for t, cut in enumerate(tables):
            reads = [[got[("schedule", i, kind, t, n), run] for n in range(len(cut))] for run in (0, 1)]
            notes = []
            results.append((spark_schedule(i, kind, [box for _, box in cut], reads, notes), notes, t))
            unread.append(lost_parts(reads, "rows", ()))
        table, notes, t = max(results, key=lambda r: schedule_score(r[0]))
        s.notes += notes
        if others := sum(unread) - unread[t]:   # a table not kept that came back blank is only another kind of table, and a lost one could be the schedule
            s.notes.append(part_note(f"{kind.capitalize()} schedule", others, sum(map(len, tables)) - len(tables[t]), i))
        if table:
            kept[i, kind] = table, t
    out = []
    for (i, kind), (table, t) in kept.items():
        rivals = [k for (j, k), (_, other) in kept.items() if j == i and k != kind and other == t]
        if rivals:
            s.notes.append(note(f"{kind.capitalize()} schedule", f"The {kind} and {rivals[0]} reads kept the same table, so neither was kept.", i))
        else:
            out.append(table | {"plans": s.picks["tags"]["spark"] if kind == "door" else []})
    return out


def spark_tags(s, crops, got):
    out = []
    for i, clips in crops["tags"].items():
        reads = both(got, "tags", i, len(clips))
        first, other = tile_points(clips, reads, "tags", "tag", s.sizes[i])
        found = [{"sheet": i, "kind": "door", "mark": m, "point": point} for m, (_, point) in first.items() if m in other]
        out += found
        if once := sorted(set(first) ^ set(other)):
            s.notes.append(note("Door tags", f"Only one read saw {some(once)}.", i))
        s.notes += plan_gaps("Door tags", reads, "tags", "tag", found, i)
    return out


def room_labels(s, crops, got):
    """Room numbers and names from plans. A plan that prints names only gives the schedule names it holds, with no number."""
    out = []
    names = [r[t["cols"][1]] for page in s.text.values() for t in page["schedules"] if t["kind"] == "finish" and len(t["cols"]) > 1 for r in t["rows"] if t["cols"][1] in r]
    for p in s.picks["rooms"]["text"]:
        found = {r["no"]: r for r in s.text[p]["rooms"]}
        out += [{"sheet": p, "number": no, "name": r["name"], "source": "text", "box": s.scale(r["box"], p)} for no, r in found.items()]
        if not found:
            printed = {norm(x["t"]) for x in s.pages[p]["lines"] if not in_boxes(x, [t["box"] for t in s.text[p]["schedules"]])}
            out += [{"sheet": p, "number": "", "name": n, "source": "text", "box": None} for n in list(dict.fromkeys(names))[:MAX_NAMES] if norm(n) in printed]
    for i, clips in crops["rooms"].items():
        reads = both(got, "rooms", i, len(clips))
        first, other = tile_points(clips, reads, "rooms", "number", s.sizes[i])
        same = {m: (field(r, "name"), point) for m, (r, point) in first.items() if m in other and room_same(field(r, "name"), field(other[m][0], "name"))}
        out += [{"sheet": i, "number": m, "name": name, "source": "spark", "box": [x - LABEL_PX, y - LABEL_PX, x + LABEL_PX, y + LABEL_PX]} for m, (name, (x, y)) in same.items()]
        if unclear := sorted((set(first) | set(other)) - set(same)):
            s.notes.append(note("Room labels", f"Two reads did not agree on {some(unclear)}.", i))
        s.notes += plan_gaps("Room labels", reads, "rooms", "number", same, i)
    return out


def table_rows(s, crops, got):
    out = []
    for i in s.ids:
        for t in s.text.get(i, {}).get("tables", ()):
            if t["rows"]:
                out.append({"sheet": i, "kind": t["kind"], "rows": [r | {"box": s.scale(r["box"], i)} for r in t["rows"]], "box": s.scale(t["box"], i)})
                if any(r.get(k, 0) is None for r in t["rows"] for k in ("area", "net_area", "clear_in", "required_in")):
                    s.notes.append(note(f"{t['kind'].capitalize()} table", "A number in this table could not be parsed.", i))
            else:
                s.notes.append(note(f"{t['kind'].capitalize()} table", "This sheet has a heading for the table, and no row of it could be read.", i))
    for i, boxes in crops["occupant"].items():
        for n, box in enumerate(boxes):
            reads = [[(num(r["area"]), num(r["factor"]), num(r["occupants"])) for r in rows_in(got[("occupant", i, n), run], "rows", ("area", "factor", "occupants")) or ()] for run in (0, 1)]
            same = [r for r in reads[0] if r in reads[1] and None not in r]
            if same:
                out.append({"sheet": i, "kind": "occupant", "rows": [{"area": a, "factor": f, "occupants": int(o)} for a, f, o in same], "box": box})
            if not same:
                s.notes.append(note("Occupant table", "The two reads gave no row they agree on for one table of this sheet.", i))
            elif any(r not in same for r in reads[0] + reads[1]):
                s.notes.append(note("Occupant table", "The two reads differ on a row of one table of this sheet.", i))
    return out


def exit_rows(s, crops, got):
    rows = []
    for i, clips in crops["exits"].items():
        row, found = exit_sheet(i, s.sizes[i], clips, both(got, "exits", i, len(clips)))
        rows.append(row)
        s.notes += found
    if not rows:
        s.notes.append(note("Exit capacity", "No life safety sheet was found in this set."))
    return {"sheets": rows}


def declared_stage(ctx) -> dict:
    """Facts for the coordination checks. See the stage contract in the build plan. Beyond it, `titles` carry `judge`: `different` when both reads of the title judge call the index title and the
    title block different drawings, `same` when one does not, None when no pair was judged. `schedules` carry `plans`, the sheets whose tags answer to them. `titles`, `index` and `rooms` carry
    a `box`, which is None where no location is known. code_fields says what `reads` and `mentions` mean.

    `cells` in `schedules` are keyed by column. A text schedule keeps its printed headings, so a finish room name is under ROOM on the Capitol and NAME on the ADU sets. A Spark schedule uses WIDTH,
    FIRE RATING, HARDWARE and ROOM for doors and NAME for finishes, and a cell its two reads differ on is None rather than a string. A number in a table row is None where the print could not be parsed.

    A row or tag is not missing just because it is not here. A `cannot_tell` row with the sheet and a `what` of Text layer, Sheet index, Door schedule, Finish schedule, Window schedule, Door tags,
    Room labels or Plan not read means that read is partial, so a partner absent there is Cannot tell and not a Fix."""
    rows = {r["index"]: r for r in ctx.stage("read").get("sheets", [])}
    sizes = {s["index"]: (s["width"], s["height"]) for s in ctx.sheets}
    paths = {s["index"]: ctx.dir / s["image"] for s in ctx.sheets}
    ids, notes = sorted(sizes), []
    floors = ctx.stage("floors")
    pages, text, picks = text_pages(ctx, ids, rows, floors, notes)
    load = functools.lru_cache(maxsize=4)(lambda i: cv2.imread(str(paths[i])))   # one image serves many tiles, and a 6000 px sheet takes a moment to open
    s = Sheets(ids, rows, sizes, pages, text, picks or select(ids, rows, text, floors), load, notes)
    reads = Reads(ctx)
    jobs, crops = first_reads(s)
    got = reads(jobs, "Reading code data, schedules and exits")
    index = sheet_index(s, crops, got)
    titles = sheet_titles(s, index, got, reads)
    schedules_read, tags = text_schedules(s)
    schedules_read += spark_schedules(s, crops, got)
    rooms_read = room_labels(s, crops, got)
    result = {"code": code_data(s, got), "index": index, "titles": titles, "tags": tags + spark_tags(s, crops, got), "schedules": schedules_read,
              "tables": table_rows(s, crops, got), "rooms": rooms_read, "exits": exit_rows(s, crops, got), "cannot_tell": notes}
    notes += not_compared(schedules_read, rooms_read)
    notes += [note("Plan not read", "The text layer of this plan could not be read, so a schedule's tags or room names on it can be missing.", p) for p in s.picks["unread"]]
    if s.picks["unplaced"]:
        names = [rows.get(i, {}).get("number") or f"page {i + 1}" for i in s.picks["unplaced"]]
        notes.append(note("Sheets not placed", f"{some(names)} could not be told apart from a code, index, schedule or life safety sheet, so they were not read for these."))
    if reads.stopped:
        notes.append(note("Spark reads", f"{reads.stopped} The reads left were not made."))
    return result


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
