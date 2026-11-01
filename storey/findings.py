"""Findings: stage facts and a jurisdiction, compared in code. No model call and no file access, so a change of jurisdiction, sprinkler status or discharge level costs nothing.

A read that did not finish is never a defect. A failed stage skips the checks that need it and says so in not_checked, and a partner that a partial read could have lost gives unknown, not Fix.
Text in a finding comes from the drawings, so it is untrusted, and the last step masks status words wherever they came from.
"""
import hashlib
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

from .declared import CODE_TITLE, LIFE_SAFETY, NOT_USED, SCHEDULE_TITLES, YEAR, floor_words, norm, num, room_same, sheet_key, some, title_same
from .floors import NOT_MAIN, _plan_sheets
from .letter import STATUS, has_status
from .rules import JURISDICTIONS, rule, stricter_when_unknown
from .vector import QUOTES
from .verticals import SHAFTS

log = logging.getLogger(__name__)

# 12 is a list for the examiner, not a finding. 15 is the second review's, which sets a state per comment.
SEVERITY = {1: "fix", 2: "fix", 3: "check", 4: "check", 5: "fix", 6: "check", 7: "fix", 8: "unknown", 9: "fix", 10: "check", 11: "check", 12: "open", 13: "check", 14: "check", 16: "check", 17: "check"}
DISCIPLINE = "A"
STAGES = ("read", "floors", "verticals", "declared", "vector")
RANK = {"fix": 0, "check": 1, "unknown": 2}
GRID = 300               # sheet px, rounded: `id` and `name` change between fresh runs and `pos` moved up to 146 px. On two bme_k runs 10 of 11 real stairs kept their cell, against 7 at 600 px and 6 with floor division, and all 11 stayed within one cell
ROUTE_MARGIN = 0.75      # routes run 16 to 19% under the architect's figures
AREA_TOLERANCE = 0.002   # rows are rounded for display: BCOE's sum 4 SF over its printed 11,538
WIDTH_TOLERANCE = 0.05   # in, a printed width against occupants times the factor
ARC_REACH_PT = 75        # tags sit under 16 pt from their doors on Capitol and Baxter, and a door with no tag 79 pt from the next tag
WALL_IN = 12             # printed corridor widths run wall centre to wall centre, 4.5 in over clear on Capitol and Baxter
NO_SWING = {"BIFOLD", "POCKET"}   # not the lab's dash too: a dash is a cased opening, and the vector stage keeps the dashed swings drawn as options, which came back on Capitol 6 and Baxter 6 and 11
STAIR_HALF = 200         # sheet px round a place with no mask, about a stair on a 6000 px sheet
TAG_HALF = 60
SIZE_CELLS = ("WIDTH", "HEIGHT", "SIZE")
MATERIALS = {"STEEL", "WOOD", "ALUMINUM", "ALUM", "HOLLOW METAL", "HM", "GLASS", "VINYL", "FIBERGLASS"}   # the whole cell only: a closed list of swings was dropped from the spec for false alarms

IBC_ONLY = ("occupancy", "construction", "occupant_load")   # the residential code asks for none of these, and only checks 11, 13 and 14 read them
CODE_FIELDS = {"edition": "code edition", "occupancy": "occupancy", "construction": "construction type", "sprinklers": "sprinkler statement", "stories": "number of stories", "area": "area", "occupant_load": "occupant load"}
OCCUPANCY_WORDS = {"ASSEMBLY": "A", "BUSINESS": "B", "EDUCATIONAL": "E", "FACTORY": "F", "HAZARD": "H", "INSTITUTIONAL": "I", "MERCANTILE": "M", "RESIDENTIAL": "R", "STORAGE": "S", "UTILITY": "U"}
NUMBER_WORDS = "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN".split()
NOTE_CHECKS = {"Sheet index": 2, "Sheet number": 2, "Text layer": 2, "Sheet title": 3, "Label under the drawing": 3, "Code summary": 4, "Occupant table": 6, "Exit table": 6, "Area table": 6, "Door schedule": 7,
               "Window schedule": 7, "Door tags": 7, "Plan not read": 7, "Exit capacity": 8, "Finish schedule": 16, "Room labels": 16}
GROUP = re.compile(r"(?<![A-Za-z0-9-])([ABEFHIMRSU])(?:-?(\d))?(?![A-Za-z0-9])")
BLOCK = re.compile(r"BLOCKS? \"?(\d[A-Z])")
SAYS_YES = re.compile(r"SPRINKLER HEADS|yes box|\bYES\b", re.I)
SAYS_NO = re.compile(r"not sprinklered|unsprinklered", re.I)
RESIDENTIAL = re.compile(r"\bIRC\b|\bCRC\b|Residential Code|one.{0,8}two.family|single.family|duplex|\bR-?3\b", re.I)
BEDROOM = re.compile(r"\bBED\s?(ROOM)?\b|\bBDRM\b", re.I)
NOT_A_STOREY = re.compile(r"foundation|roof|attic|crawl", re.I)
BELOW_GRADE = re.compile(r"basement|cellar", re.I)
SITE_TITLE = re.compile(r"\b(?:site|plot|civil)\b", re.I)
LENGTH = re.compile(r"(?:(\d+)\s*')?\s*-?\s*(?:(\d+(?:\s+\d+/[1-9]\d*)?)\s*\")?")   # 4'-0", 4' - 0 1/2", 4' or 11", anywhere in a cell
PLACED = re.compile(r"(\w+)@(-?\d+),(-?\d+)(?:#\d+)?")
MASK = "[word removed]"
_mark = sheet_key        # marks normalise as sheet numbers do, so `1 a`, `1A` and `1-A` are one mark

OTHER_TRADES = "Electrical, plumbing, mechanical, structural, zoning and energy sheets were not reviewed."
DWELLING = ("This set shows a one- or two-family dwelling, which uses the International Residential Code, and this product does not check that code. These were not checked: "
            "a life safety sheet, exits per floor, travel distance, corridor width, the code edition against the adopted code, and the code summary's occupancy, construction type and occupant load.")
ENDS_LIMIT = "A stair that ends on its top floor looks the same as one erased from that floor."
CONTINUOUS = "A stair stays continuous between the floors it serves."
REACHES = "A stair that serves upper floors reaches the level of exit discharge."
SEVERAL_PLANS = "{} holds several floor plans, and only the main one was read, so the others were not followed."   # true on a sheet outside the floor order too, such as Abbot A2.0, where the read found no floor name
LEGEND = {"missing": "Missing", "end": "Not on this floor", "check": "Check"}   # the screen's words, because a state name means nothing in a letter
ONE_NAME = "One name for each room"
VECTOR = "Vector geometry."
ROUTE_EVIDENCE = "Vector route, shown as a minimum, because routes run 16 to 19% under architects' figures."


def _groups(value):
    """The IBC groups in an occupancy as printed, one per item, because a rule row reads one group at a time."""
    # shortcut: a capital A opening a phrase reads as group A, match against the printed code's group list if a real value does this.
    found = [f"{letter}-{digit}" if digit else letter for letter, digit in GROUP.findall(value)]
    return found or [letter for word, letter in OCCUPANCY_WORDS.items() if word in norm(value)]


def _story_count(value):
    """Stories as printed, in digits or words, or None."""
    return num(value) or next((NUMBER_WORDS.index(word) + 1 for word in norm(value).split() if word in NUMBER_WORDS), None)


def _key(check, number, thing, place=None):
    digest = hashlib.sha256(f"{check}|{number}|{thing}".encode()).hexdigest()[:16]
    return f"{digest}@{place[0]},{place[1]}" if place else digest


def _cell(point):
    return [round(v / GRID) for v in point] if point else None


def _place(row, cell):
    """A stair's grid cell. Never `id`, which is cluster order and changes between fresh runs."""
    # shortcut: the grid is in sheet px, measured on 6000 px sheets, so a smaller sheet gets a coarser key, scale it if evaluate is given sheet sizes.
    return _cell(row.get("pos") or cell.get("point"))


def _around(point, half):
    return [point[0] - half, point[1] - half, point[0] + half, point[1] + half] if point else None


def _union(boxes):
    boxes = [b for b in boxes if b]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)] if boxes else None


def _marks(texts):
    return some(sorted(set(texts), key=lambda mark: (len(mark), mark)))


def _source(item):
    return "Text layer." if item.get("source") == "text" else "Two Spark reads."


def _cell_box(cell):
    return cell.get("box") or _around(cell.get("point"), STAIR_HALF)


def _evidence(cell):
    signals = cell.get("signals") or {}
    if cell.get("crop"):
        crop = f"Crop {cell['crop']}"
    elif signals.get("aligned") is False:
        crop = "No crop, because this floor is not aligned, so the place is not known"
    else:
        crop = "No crop, because nothing is drawn at this place"
    read = ", ".join(f"{label} {signals[name]}" for label, name in (("Runs", "spark"), ("crop answer", "crop"), ("ink", "ink")) if signals.get(name) is not None)
    return ". ".join(filter(None, (crop, read, signals.get("reason")))) + "."


def _crops(cells):
    names = [c["crop"] for c in cells if c.get("crop")]
    return f"Crops {', '.join(names)}." if names else "No crop was cut."


def _lookup(id, jurisdiction, **inputs):
    """The rule row, or None if no case fits."""
    try:
        return rule(id, jurisdiction, **inputs)
    except KeyError:
        return None


def _section(row):
    return f"{row['section']}, {row['adopted']}"


def _required(text, row):
    return " ".join([text, *row.get("conditions", ()), *(["Examiner to confirm."] if row.get("confirm") else [])])


def _strictest(rows, pick):
    """The strictest of one row per declared group, or None if a group has no row or no group was read. NP is stricter than any number."""
    if not rows or None in rows:
        return None
    return next((row for row in rows if row["value"] == "NP"), None) or pick(rows, key=lambda row: row["value"])


@dataclass
class Review:
    """One evaluate call: the stage facts, what the examiner chose, and what the checks raised. A stage that did not finish is an empty dict here."""
    read: dict
    floors: dict
    verticals: dict
    declared: dict
    vector: dict
    jurisdiction: str
    sprinklers: str | None
    discharge: int | None
    findings: list = field(default_factory=list)
    open_items: list = field(default_factory=list)
    stricter: bool = False

    def __post_init__(self):
        self.sheets = {s["index"]: s for s in self.read.get("sheets", ())}
        self.order = self.floors.get("order") or []
        self.code = self.declared.get("code") or {}
        self.notes = self.declared.get("cannot_tell") or []
        self.groups = _groups(self.value("occupancy"))
        self.load = int(n) if (n := num(self.value("occupant_load"))) else None
        self.sprinklers = next((v for v in (self.sprinklers, self.value("sprinklers")) if v in ("yes", "no")), None)
        chosen = self.floors.get("discharge") if self.discharge is None else self.discharge
        self.discharge = chosen if chosen in self.order else None
        self.dwelling = self._dwelling()

    def _dwelling(self):
        # shortcut: with no occupancy printed, or no code sheet, a bedroom label reads as a dwelling, so a dormitory moves to Not checked, ask the examiner if a real set does this.
        if set(self.groups) - {"R", "R-3", "U"}:
            return False   # a declared IBC occupancy outranks room labels: an apartment block has bedrooms too. A bare R says no more than the word residential
        if RESIDENTIAL.search(f"{self.value('occupancy')} {self.value('edition')}"):
            return True
        settled = not self.code or (self.code.get("occupancy") or {}).get("reads", 0) >= 2   # a lost read says nothing about what is printed
        return settled and not self.groups and any(BEDROOM.search(x.get("name") or "") for x in self.declared.get("rooms") or ())

    def value(self, name):
        return (self.code.get(name) or {}).get("value") or ""

    def number(self, i):
        return self.sheets.get(i, {}).get("number") or f"page {i + 1}"

    def floor(self, i):
        return self.sheets.get(i, {}).get("level") or self.number(i)

    def sprinkler_row(self, id, **inputs):
        """The rule row at this review's sprinkler status. With none, the stricter row, and the review says so."""
        try:
            if self.sprinklers:
                return rule(id, self.jurisdiction, sprinklers=self.sprinklers, **inputs)
            self.stricter = True
            return stricter_when_unknown(id, self.jurisdiction, **inputs)
        except KeyError:
            return None

    def partial(self, whats, sheets=None):
        """True if a read a missing partner lives in did not finish, so its absence is unknown, not a Fix. See declared_stage."""
        return any(n["what"] in (*whats, "Text layer", "Spark reads") and (sheets is None or n.get("sheet") is None or n["sheet"] in sheets) for n in self.notes)

    def add(self, check, title, *, sheet=None, box=None, severity=None, measured="", required="", section="", evidence="", vertical=None, thing=None, place=None):
        self.findings.append({"key": _key(check, "" if sheet is None else self.number(sheet), title if thing is None else thing, place), "check": check, "severity": severity or SEVERITY[check], "discipline": DISCIPLINE,
                              "sheet": sheet, "box": box, "title": title, "measured": measured, "required": required, "rule": section, "evidence": evidence, "vertical": vertical})

    def stair(self, check, severity, row, i, cell, measured, required, section=""):
        words = "is not confirmed on" if cell.get("state") == "check" else "stops above" if check == 10 else "is missing on"
        if not self.floors.get("order_read"):   # a Fix needs direct evidence, and here which floors adjoin rests on a reply that failed
            severity, measured = "check", f"{measured} The floors are in sheet order, because their order could not be read."
        self.add(check, f"Stair {row['id']} {words} {self.floor(i)}", sheet=i, box=_cell_box(cell), severity=severity, measured=measured, required=required, section=section,
                 evidence=_evidence(cell), vertical=row["id"], thing="stair", place=_place(row, cell))

    def item(self, title, sheet, row, cell, evidence):
        self.open_items.append({"key": _key(12, "" if sheet is None else self.number(sheet), "stair", _place(row, cell)), "check": 12, "sheet": sheet, "box": _cell_box(cell), "title": title, "evidence": evidence, "vertical": row["id"]})


def _required_sheets(r):
    rows = list(r.sheets.values())
    titles = [x["title"] for x in (*rows, *(r.declared.get("titles") or ())) if x.get("title")]
    schedules = r.declared.get("schedules") or ()

    def titled(pattern):
        return any(pattern.search(t) for t in titles)

    present = {"code summary": any((row or {}).get("value") for row in r.code.values()) or titled(CODE_TITLE) or any(s.get("cls") == "code_summary" for s in rows),
               "door schedule": any(s["kind"] == "door" for s in schedules) or titled(SCHEDULE_TITLES["door"]) or any(n["what"] == "Door schedule" for n in r.notes),
               "floor plan": any(s.get("cls") == "floor_plan" for s in rows),
               "site plan": titled(SITE_TITLE) or any(s.get("cls") == "site_plan" for s in rows)}
    if not r.dwelling:
        present["life safety plan"] = bool((r.declared.get("exits") or {}).get("sheets")) or titled(LIFE_SAFETY)
    unsure = not all(s.get("read_ok") for s in rows) or r.partial(("Sheets not placed",))
    for name, there in present.items():
        if not there:
            severity = "unknown" if unsure else "check" if name == "site plan" else None   # an interior alteration can properly have no site plan
            r.add(1, f"No {name} in the set", severity=severity, measured="No sheet was read as one.", required=f"A {name}", evidence="Sheet read.", thing=name)


def _unread_sheets(r):
    """A sheet the read lost is in no floor order, so the floors either side of it read as next to each other."""
    for i, s in r.sheets.items():
        if not s.get("read_ok"):
            r.add(1, "This sheet could not be read", sheet=i, severity="unknown", measured="The read gave no number, title or class, so this sheet is outside the floor order. If it is a floor plan, its stairs, travel distance and corridor width were not checked.",
                  required="Examiner to review this sheet", evidence="Sheet read.", thing="unread")


def _sheet_index(r):
    index, titles = r.declared.get("index") or [], r.declared.get("titles") or []
    if not index:
        return   # declared notes it, and with no index every sheet would read as unlisted
    rows, sheets = defaultdict(list), defaultdict(list)
    for x in index:
        if not x.get("not_used"):
            rows[sheet_key(x["number"])].append(x)
    for t in titles:
        if t.get("number"):
            sheets[sheet_key(t["number"])].append(t)
    blank = not all(t.get("number") for t in titles)
    for key, group in rows.items():
        x = group[0]
        number = x["number"]
        if key not in sheets:
            r.add(2, f"Sheet {number} is on the index and not in the set", sheet=x["sheet"], box=x.get("box"), severity="unknown" if blank else None, measured=f"The index lists {number}, {x['title']}.", required="A sheet for each index row",
                  evidence="Sheet index.", thing=f"row {key}")
        if len(group) > 1:
            r.add(2, f"Sheet {number} is listed {len(group)} times on the index", sheet=x["sheet"], box=_union(y.get("box") for y in group), severity="check", measured="; ".join(y["title"] for y in group), required="One index row for each sheet",
                  evidence="Sheet index.", thing=f"twice {key}")
    listed = {sheet_key(x["number"]) for x in index}
    lost = r.partial(("Sheet index",), {x["sheet"] for x in index})
    for key, group in sheets.items():
        t = group[0]
        number = t["number"]
        if key not in listed:
            r.add(2, f"Sheet {number} is in the set and not on the index", sheet=t["sheet"], box=t.get("box"), severity="unknown" if lost else None, measured=f"{number} is titled {t['title']}.", required="An index row for each sheet", evidence=_source(t), thing=f"sheet {key}")
        if len(group) > 1:
            r.add(2, f"{len(group)} sheets carry the number {number}", sheet=t["sheet"], box=t.get("box"), severity="check", measured="; ".join(y["title"] for y in group), required="One number for each sheet", evidence=_source(t), thing=f"number {key}")


def _blocks(text):
    return set(BLOCK.findall(text.upper()))


def _differ(read, a, b):
    """Both name a floor or block, and no name is shared. A title with neither word says nothing."""
    x, y = read(a), read(b)
    return bool(x and y and not x & y)


def _titles(r):
    rows = defaultdict(list)
    for x in r.declared.get("index") or ():
        rows[sheet_key(x["number"])].append(x["title"])
    for t in r.declared.get("titles") or ():
        name, under = t.get("title") or "", t.get("under_drawing") or ""
        listed = rows.get(sheet_key(t.get("number") or ""), ())
        on_label = _differ(floor_words, name, under)
        on_index = bool(listed) and not any(title_same(x, name) for x in listed) and (t.get("judge") == "different" or any(_differ(floor_words, x, name) or _differ(_blocks, x, name) for x in listed))
        if on_label or on_index:
            said = [f"Title block says {name}.", *([f"The drawing says {under.split('; ')[0]}."] if on_label else []), *([f"The index says {listed[0]}."] if on_index else [])]
            r.add(3, "Floor name differs" if on_label else "Sheet title differs from the index", sheet=t["sheet"], box=t.get("box"), measured=" ".join(said), required="One name for this sheet", evidence=_source(t), thing="title")


def _code_summary(r):
    for name, label in CODE_FIELDS.items():
        row = r.code.get(name)
        if row is None or (r.dwelling and name in IBC_ONLY):
            continue
        value, where = row.get("value"), {"sheet": row.get("sheet"), "box": row.get("box"), "evidence": "Code sheet."}
        if value and name == "edition" and not YEAR.search(value):
            r.add(4, "The code edition has no year", measured=value, required="The code and its year", thing=name, **where)
        elif not value and row.get("reads", 0) >= 2:
            r.add(4, f"The code summary has no {label}", required=f"The {label}", thing=name, **where)
    sprinklers = r.code.get("sprinklers") or {}
    against = {"no": SAYS_YES, "yes": SAYS_NO}.get(sprinklers.get("value"))
    other = next((m for m in sprinklers.get("mentions") or () if against and against.search(m)), None)
    if other:
        r.add(4, "Sprinkler statements disagree", sheet=sprinklers.get("sheet"), box=sprinklers.get("box"), measured=f"The sheet says {sprinklers['value']}, and also says: {other}", required="One sprinkler statement", evidence="Code sheet.", thing="sprinkler statements")


def _left_out(r):
    """Plan sheets outside the order that can be a storey. A life safety or demolition copy of a named floor is left out on purpose."""
    return [s for s in _plan_sheets(list(r.sheets.values())) if s["index"] not in r.order and not NOT_A_STOREY.search(s["title"]) and not (s["level"].strip() and NOT_MAIN.search(s["title"]))]


def _plans_left_out(r):
    names = [r.number(s["index"]) for s in _left_out(r)]
    if names:
        r.add(1, f"Stairs, travel distance and corridor width were not checked on {some(names)}", severity="unknown", measured=f"Outside the floor order: {', '.join(names)}.", required="One floor plan in the floor order for each storey",
              evidence="Sheet read and floor order.", thing="left out")


def _stories(r):
    row = r.code.get("stories") or {}
    declared = _story_count(row.get("value") or "")
    plans = [r.floor(i) for i in r.order if not NOT_A_STOREY.search(r.floor(i))]
    above = sum(not BELOW_GRADE.search(p) for p in plans)
    if declared is None or not plans or declared in (above, len(plans)):
        return
    unsure = _left_out(r) or not all(s.get("read_ok") for s in r.sheets.values())   # a plan outside the order, or a sheet not read, can be the storey that looks missing
    r.add(5, "Stories declared differ from the floor plans", sheet=row.get("sheet"), box=row.get("box"), severity="unknown" if unsure else None, measured=f"The code summary declares {declared:g}. The set has {above} floor plans above grade.",
          required="One floor plan per storey", evidence="Code sheet and floor order.", thing="stories")


def _edition(r):
    row = r.code.get("edition") or {}
    years = [m[0] for m in YEAR.finditer(row.get("value") or "")]
    adopted = JURISDICTIONS[r.jurisdiction]
    if years and not any(year in adopted for year in years):
        r.add(5, "Code edition differs from the adopted code", sheet=row.get("sheet"), box=row.get("box"), measured=row["value"], required=adopted, evidence="Code sheet.", thing="edition")


def _load_table(r, t):
    rooms = [x for x in t["rows"] if not x.get("total") and "note" not in x]
    for x in rooms:
        area, factor, printed = x.get("area"), x.get("factor"), x.get("occupants")
        if area and factor and printed is not None:
            worked = math.ceil(area / factor - 1e-9)
            if worked != printed:
                r.add(6, f"Room {x['room']} occupant load" if x.get("room") else "Occupant load row", sheet=t["sheet"], box=x.get("box") or t.get("box"), thing=f"load {x.get('room') or (area, factor)}", required="The area over the factor, rounded up",
                      measured=f"{area:g} SF at {factor:g} SF per occupant gives {worked}, and the table says {printed}. " + ("A load rounded down is unsafe." if printed < worked else "A load over-declared is safe."), evidence="Occupant table.")
    total = next((x for x in t["rows"] if x.get("total")), None)
    if total:
        printed, summed = total.get("occupants"), sum(x.get("occupants") or 0 for x in rooms)
        if printed is not None and printed != summed:
            r.add(6, "Occupant total differs from its rows", sheet=t["sheet"], box=total.get("box"), thing="load total", required="A total that matches its rows", evidence="Occupant table.",
                  measured=f"Rows sum to {summed}, and the total says {printed}. " + ("A load rounded down is unsafe." if printed < summed else "A load over-declared is safe."))
        _area_total(r, t, rooms, total, "Area total")
    note = next((x for x in t["rows"] if "note" in x), None)
    if note and any(x.get("factor") for x in rooms) and None not in (note.get("area"), note.get("net_area")):
        loadless = sum(x.get("area") or 0 for x in rooms if x.get("factor") is None)
        worked = note["area"] - loadless
        if abs(worked - note["net_area"]) > AREA_TOLERANCE * note["net_area"]:
            r.add(6, "Net area in the note differs from the table", sheet=t["sheet"], box=note.get("box"), thing="net area", required="A note that matches the table", evidence="Occupant table.",
                  measured=f"The note gives {note['net_area']:,.0f} SF. {note['area']:,.0f} SF less {loadless:,.0f} SF of rows with no load gives {worked:,.0f} SF.")


def _area_total(r, t, rows, total, title):
    areas = [x.get("area") for x in rows]
    printed = total.get("area")
    if printed and None not in areas and abs(sum(areas) - printed) > AREA_TOLERANCE * printed:
        r.add(6, title + " differs from its rows", sheet=t["sheet"], box=total.get("box"), measured=f"Rows sum to {sum(areas):,.0f} SF, and the total says {printed:,.0f} SF.", required="A total that matches its rows", evidence=f"{t['kind'].capitalize()} table.", thing=f"{t['kind']} area total")


def _worked_row(rows, example):
    """The table row a worked example repeats, by name, or by the number in `CORRIDOR 140` against `140 SECURE CONFERENCE CIRCULATION`."""
    name = example["exit"]
    return next((x for x in rows if x["exit"] == name or name.split()[-1:] == x["exit"].split()[:1]), None)


def _width_note(printed, worked, occupants, alarmed):
    """What a printed width means against the worked figure. `alarmed` holds the factors that also need a voice alarm."""
    if printed > worked:
        return "A width above the worked figure is safe."
    factor = next((f for f in sorted(alarmed) if abs(printed - occupants * f) <= WIDTH_TOLERANCE), None)
    if factor:
        return f"That is the {factor} in per occupant row, which also needs a voice alarm. No voice alarm was read, so the examiner must confirm this value."
    return "A width below the worked figure is unsafe."


def _exit_table(r, t, tables):
    door, stair = rule("ibc.1005.3.2", r.jurisdiction), rule("ibc.1005.3.1", r.jurisdiction)
    # no stage reads a voice alarm, so its row changes how an item reads and never clears one: a fact nobody read must not unlock a more lenient case
    alarmed = {rule("ibc.1005.3.2", r.jurisdiction, occupancy=g, sprinklers="yes", voice_alarm="yes")["value"] for g in r.groups} if r.sprinklers == "yes" else ()
    rows = [x for x in t["rows"] if not x.get("example")]
    for x in rows:
        name, clear, occupants, printed = x["exit"], x.get("clear_in"), x.get("occupants"), x.get("required_in")
        if occupants is None or printed is None:
            continue
        worked = occupants * door["value"]
        if abs(printed - worked) > WIDTH_TOLERANCE and abs(printed - occupants * stair["value"]) > WIDTH_TOLERANCE:
            r.add(6, f"{name} required width", sheet=t["sheet"], box=x.get("box"), section=_section(door), thing=f"width {name}", required=f"{door['value']} in per occupant", evidence="Exit table.",
                  measured=f"{occupants} occupants give {worked:g} in, and the table says {printed:g} in. " + _width_note(printed, worked, occupants, alarmed))
        if clear is not None and clear < printed:
            r.add(6, f"{name} clear width is below its required width", sheet=t["sheet"], box=x.get("box"), measured=f"{clear:g} in clear", required=f"{printed:g} in", evidence="Exit table.", thing=f"clear {name}")
    for example in (x for x in t["rows"] if x.get("example")):
        row = _worked_row(rows, example)
        if row and row.get("occupants") != example.get("occupants"):
            r.add(6, f"Worked example for {example['exit']} differs from the table", sheet=t["sheet"], box=example.get("box"), measured=f"The example uses {example.get('occupants')} occupants, and the table says {row.get('occupants')}.", required="One figure in the example and the table",
                  evidence="Exit table.", thing=f"example {example['exit']}")
    exits = [x for x in rows if x["exit"].startswith("EXIT ")]
    total = next((x for o in tables if o["kind"] == "occupant" and o["sheet"] == t["sheet"] for x in o["rows"] if x.get("total")), None)
    if exits and total and total.get("occupants") is not None:
        carried, printed = sum(x.get("occupants") or 0 for x in exits), total["occupants"]
        if carried != printed:
            r.add(6, "Exit table does not carry the occupant total", sheet=t["sheet"], box=_union(x.get("box") for x in exits), thing="exit sum", required="An exit table that carries the occupant total", evidence="Exit and occupant tables.",
                  measured=f"The exits carry {carried}, and the occupant total is {printed}. " + ("A load over-declared is safe." if carried > printed else "A load rounded down is unsafe."))


def _tables(r):
    tables = r.declared.get("tables") or []
    for t in tables:
        if t["kind"] == "occupant":
            _load_table(r, t)
        elif t["kind"] == "exit":
            _exit_table(r, t, tables)
        elif t["kind"] == "area":
            total = next((x for x in t["rows"] if x.get("total")), None)
            if total:
                _area_total(r, t, [x for x in t["rows"] if not x.get("total")], total, "Area total")


def _tag_schedules(r):
    return [s for s in r.declared.get("schedules") or () if s["kind"] != "finish"]


def _tags(r):
    schedules, tags = _tag_schedules(r), r.declared.get("tags") or []

    def unsure(kind, sheets):
        return "unknown" if r.partial((f"{kind} schedule", "Door tags", "Plan not read"), sheets) else None

    for s in schedules:
        plans, kind = s.get("plans") or [], s["kind"].capitalize()
        seen = Counter(_mark(t["mark"]) for t in tags if t["kind"] == s["kind"] and t["sheet"] in plans)
        if not seen:
            continue   # declared notes a schedule that nothing on a plan answers to
        alone = []
        for row in s["rows"]:
            mark = _mark(row["mark"])
            quantity = num(row["cells"]["QTY"]) if row["cells"].get("QTY") else None
            if not seen[mark] and quantity != 0:
                alone.append(row)
            elif quantity is not None and seen[mark] != quantity:
                r.add(7, f"{kind} {row['mark']} count differs from the schedule", sheet=s["sheet"], box=row.get("box"), severity="check", measured=f"The schedule says {quantity:g}, and the plan shows {seen[mark]} tags.", required="One tag for each counted item",
                      evidence=_source(s), thing=f"count {s['kind']} {mark}")
        if alone:
            r.add(7, f"{kind} schedule rows with no tag on the plan: {_marks(row['mark'] for row in alone)}", sheet=s["sheet"], box=_union(row.get("box") for row in alone), severity=unsure(kind, {s["sheet"], *plans}),
                  measured=f"{len(alone)} of {len(s['rows'])} rows", required="A tag on the plan for each schedule row", evidence=_source(s), thing=f"rows {s['kind']}")
    for kind, plan in dict.fromkeys((s["kind"], plan) for s in schedules for plan in s.get("plans") or ()):
        serving = [s for s in schedules if s["kind"] == kind and plan in (s.get("plans") or ())]
        known = {_mark(row["mark"]) for s in serving for row in s["rows"]}
        here = {t["mark"] for t in tags if t["kind"] == kind and t["sheet"] == plan}
        extra = [t for t in tags if t["kind"] == kind and t["sheet"] == plan and _mark(t["mark"]) not in known]
        if extra:
            name = kind.capitalize()
            r.add(7, f"{name} tags with no schedule row: {_marks(t['mark'] for t in extra)}", sheet=plan, box=_union(_around(t["point"], TAG_HALF) for t in extra), severity=unsure(name, {plan, *(s["sheet"] for s in serving)}),
                  measured=f"{len({t['mark'] for t in extra})} of {len(here)} tags on this plan", required="A schedule row for each tag", evidence=_source(serving[0]), thing=f"tags {kind}")


def _swings(r):
    """Each arc goes to its nearest tag, and a tag keeps its nearest arc."""
    # shortcut: the vector stage does not say which arcs are dashed, so an option drawn on a bifold or pocket row reads as its door, have it report them if one shows up.
    sheets, tags = r.vector.get("sheets") or {}, r.declared.get("tags") or []
    for s in (s for s in _tag_schedules(r) if s["kind"] == "door"):
        rows = {_mark(row["mark"]): row for row in s["rows"]}
        for plan in s.get("plans") or ():
            entry = sheets.get(str(plan)) or {}
            ipp = entry.get("scale_in_per_pt")
            here = [t for t in tags if t["kind"] == "door" and t["sheet"] == plan and _mark(t["mark"]) in rows]
            nearest = {}
            for door in (entry.get("doors") or []) if ipp and here else ():
                px_per_pt = 2 * math.dist(door["hinge"], door["point"]) / door["width_in"] * ipp   # the leaf's px over its inches, because evaluate has no sheet sizes
                gap, n = min((math.dist(door["point"], t["point"]), n) for n, t in enumerate(here))
                if gap < ARC_REACH_PT * px_per_pt and (n not in nearest or gap < nearest[n][0]):
                    nearest[n] = gap, door
            for n, (_, door) in nearest.items():
                t = here[n]
                cells = rows[_mark(t["mark"])]["cells"]
                swing = cells.get("SWING") or ""
                if swing.upper() in NO_SWING:
                    printed = " ".join(filter(None, (swing, cells.get("SIZE"))))
                    r.add(7, f"Door {t['mark']} type differs from the plan", sheet=plan, box=_union([_around(t["point"], TAG_HALF), _around(door["point"], TAG_HALF)]), severity="check", measured=f"The schedule says {printed}, and the plan shows a swing arc of {door['width_in']:g} in.",
                          required="One door type for each tag", evidence=f"{_source(s)} {VECTOR}", thing=f"arc {t['mark']}")


def _size(text, column):
    """The lengths a size cell prints, in inches: the first, or the first two in a SIZE cell, which prints width by height. A footnote mark, a remark or a merged column after them is no part of the size."""
    lengths = [12 * int(feet or 0) + sum(map(Fraction, inches.split())) for feet, inches in LENGTH.findall(text.translate(QUOTES)) if feet or inches]
    return tuple(lengths[:2 if column == "SIZE" else 1])


def _window_types(r):
    """A window mark that is a letter names a type, so its size holds across the set whatever the quantity. Door marks restart per floor, so this is windows only."""
    # shortcut: a set of two modules could use one letter for two types, though Abbot's modules agree, keep to sheets of one letter as _rooms_near does if a set does this.
    rows = defaultdict(list)
    for s in (s for s in _tag_schedules(r) if s["kind"] == "window"):
        for row in s["rows"]:
            if re.fullmatch("[A-Z]", _mark(row["mark"])):
                rows[_mark(row["mark"])].append((s, row))
    for mark, group in rows.items():
        for column in SIZE_CELLS:
            read = [(s, row, _size(row["cells"][column], column)) for s, row in group if norm(row["cells"].get(column))]
            lengths = any(size for *_, size in read)   # a cell with no length is compared as text only when no cell in the group has one
            sizes = defaultdict(list)
            for s, row, size in read:
                if size or not lengths:
                    sizes[size or norm(row["cells"][column])].append((s, row))
            if len(sizes) > 1:
                odd, *rest = sorted(sizes.values(), key=len)
                said = ", and ".join(f"{found[0][1]['cells'][column]} on {' and '.join(r.number(s['sheet']) for s, _ in found)}" for found in (odd, *rest))
                r.add(7, f"Window {mark} has more than one size", sheet=odd[0][0]["sheet"], box=odd[0][1].get("box"), severity="check", measured=f"{column} reads {said}.", required="One size for each window type",
                      evidence=_source(odd[0][0]), thing=f"size {mark} {column}")


def _swing_materials(r):
    for s in (s for s in _tag_schedules(r) if s["kind"] == "door"):
        for row in s["rows"]:
            swing = row["cells"].get("SWING")
            if norm(swing) in MATERIALS:
                r.add(7, f"Door {row['mark']} has a material in the swing column", sheet=s["sheet"], box=row.get("box"), severity="check", measured=f"The swing column says {swing.strip()}.", required="A swing in the swing column",
                      evidence=_source(s), thing=f"swing {_mark(row['mark'])}")


def _exit_capacity(r):
    for s in (r.declared.get("exits") or {}).get("sheets") or ():
        total = s.get("sum")
        if total is None:
            continue   # declared notes why no sum stands
        title, measured, evidence = "Exit capacity on this sheet", f"Exit markers sum to {total} people on two equal reads.", "Two Spark reads under the half-sheet rule."
        # shortcut: the printed load is the building's, so one floor's sheet can read low, compare per floor if a stage reads floor loads.
        if r.load and total < r.load:
            r.add(8, title, sheet=s["sheet"], severity="check", measured=measured, required=f"At least the printed load of {r.load}", evidence=evidence, thing="sum")
        else:
            r.add(8, title, sheet=s["sheet"], measured=measured, required=f"Examiner to confirm against the printed load of {r.load}" if r.load else "A printed occupant load to compare with", evidence=evidence, thing="sum")


def _established(r):
    """Each stair confirmed by two runs, with its cells bottom to top and the positions of its Found floors."""
    for row in r.verticals.get("verticals") or ():
        if row.get("established"):
            cells = [row["floors"].get(str(i)) or {} for i in r.order]
            yield row, cells, [k for k, c in enumerate(cells) if c.get("state") == "found"]


def _aligned(r, i, cell):
    return (cell.get("signals") or {}).get("aligned") is not False and i not in (r.floors.get("unaligned") or ())


def _stair_gaps(r):
    for row, cells, found in _established(r):
        for k, cell in enumerate(cells):
            state = cell.get("state")
            if state in ("missing", "check") and found and found[0] < k < found[-1]:
                i = r.order[k]
                below, above = r.floor(r.order[max(f for f in found if f < k)]), r.floor(r.order[min(f for f in found if f > k)])
                here = f"No stair at this place on {r.floor(i)}." if state == "missing" else f"{(cell.get('signals') or {}).get('reason')} on {r.floor(i)}."
                r.stair(9, "fix" if state == "missing" and _aligned(r, i, cell) else "check", row, i, cell, f"Found on {below} and {above}. {here}", CONTINUOUS)


def _stair_discharge(r):
    rows = list(_established(r))
    if r.discharge is None:
        if rows:
            r.add(10, "The level of exit discharge is not known", severity="unknown", measured="No floor label settles it, so no stair was compared with it.", required="Examiner to choose the level", evidence="Floor labels.", thing="discharge")
        return
    d = r.order.index(r.discharge)
    section = _section(rule("ibc.1023.3", r.jurisdiction))
    for row, cells, found in rows:
        cell = cells[d]
        state = cell.get("state")
        if found and found[-1] > d and state in ("missing", "end", "check"):
            bottom, top = r.floor(r.order[found[0]]), r.floor(r.order[found[-1]])
            seen = f"Found on {top}" if bottom == top else f"Found from {bottom} to {top}"
            r.stair(10, "check", row, r.discharge, cell, f"{seen}. On {r.floor(r.discharge)}, the level of exit discharge, it reads {LEGEND[state]}.", REACHES, section)


def _single_exit(r, story):
    """The rows that let this story use one exit, or None. A group or a load nobody read unlocks nothing."""
    rows = []
    for group in r.groups:
        row = r.sprinkler_row("ibc.1006.3.4.t1", story=story) if group == "R-2" else _lookup("ibc.1006.3.4.t2", r.jurisdiction, occupancy=group, story=story)
        if row is None or row["value"] == "NP" or (group != "R-2" and (r.load is None or r.load > row["value"])):
            return None
        rows.append(row)
    return rows or None


def _short_floors(r, rows, table):
    """(sheet, found cells, cells to confirm, the rows that let it use one exit or None) for each storey above discharge with fewer found stairs than its load needs."""
    above = r.order[r.order.index(r.discharge) + 1:]
    for story, i in enumerate((i for i in above if not NOT_A_STOREY.search(r.floor(i))), 2):
        cells = [row["floors"].get(str(i)) or {} for row in rows]
        found, unsure = [c for c in cells if c.get("state") == "found"], [c for c in cells if c.get("state") == "check"]
        if len(found) < table["value"]:
            yield i, found, unsure, _single_exit(r, story) if found else None


def _exits_per_floor(r):
    """Floors above discharge only, where every exit is a stair. On the discharge floor exits are doors, which are never inferred from a plan."""
    # shortcut: a basement's stairs are not compared, compare them if a stage reads loads per floor.
    rows = [row for row, *_ in _established(r)]
    if not rows:
        return
    table = _lookup("ibc.1006.3.3", r.jurisdiction, occupant_load=r.load)
    if table is None or r.discharge is None:
        why = "No occupant load is printed." if table is None else "The level of exit discharge is not known."
        r.add(11, "Exits per floor were not compared", severity="unknown", measured=why, required="Stairs against the exits the load requires", evidence="Stair matrix.", thing="exits")
        return
    conditional = []
    for i, found, unsure, allowed in _short_floors(r, rows, table):
        if allowed is None:
            r.add(11, f"Too few stairs on {r.floor(i)}", sheet=i, severity="check", measured=f"{len(found)} found and {len(unsure)} to confirm.", required=f"{table['value']} exits for a load of {r.load}", section=_section(table), evidence=_crops([*found, *unsure]), thing="exits")
        elif any(row.get("confirm") for row in allowed):
            conditional.append((i, allowed, found))
    if conditional:
        rows = [row for _, allowed, _ in conditional for row in allowed]
        terms = dict.fromkeys(c for row in rows for c in row.get("conditions", ()))
        sections = dict.fromkeys(row["section"] for row in rows)
        r.add(11, "One stair here depends on conditions", severity="unknown", measured=f"One stair on {some(r.floor(i) for i, _, _ in conditional)}, against {table['value']} exits for a load of {r.load}.", required=_required("One exit is allowed only if:", {"conditions": terms, "confirm": True}),
              section=f"{', '.join(sections)}, {rows[0]['adopted']}", evidence=_crops(c for _, _, found in conditional for c in found), thing="single exit")


def _stair_list(r):
    d = r.order.index(r.discharge) if r.discharge is not None else None
    for row, cells, found in _established(r):
        for k, cell in enumerate(cells):
            state, i = cell.get("state"), r.order[k]
            raised = bool(found) and (found[0] < k < found[-1] or (k == d and found[-1] > d))
            if state == "check" and not raised:
                r.item(f"Stair {row['id']} is not confirmed on {r.floor(i)}, at the end of its run", i, row, cell, _evidence(cell))
            elif state == "missing" and not raised:
                r.item(f"Stair {row['id']} reads missing on {r.floor(i)}, next to a floor that is not confirmed", i, row, cell, _evidence(cell))
        if found:
            low, high = cells[found[0]], cells[found[-1]]
            bottom, top = r.order[found[0]], r.order[found[-1]]
            title = f"Stair {row['id']} runs from {r.floor(bottom)} to {r.floor(top)}" if bottom != top else f"Stair {row['id']} is found on {r.floor(top)} only"
            r.item(title, top, row, high, f"{_crops([low, high])} {ENDS_LIMIT}")
        else:
            r.item(f"Stair {row['id']} is found on no floor", None, row, cells[0] if cells else {}, f"{_crops(cells)} {ENDS_LIMIT}")
    for row in r.verticals.get("verticals") or ():
        cells = list((row.get("floors") or {}).items())
        if not row.get("established") or any(c.get("state") == "not_followed" for _, c in cells):
            seen = next(((int(i), c) for i, c in cells if c.get("state") == "check"), None)
            r.item(f"Stair {row['id']} was seen by one run only, so it was not followed", seen and seen[0], row, seen[1] if seen else {}, _crops(c for _, c in cells))


def _floors_measured(r):
    sheets = r.vector.get("sheets") or {}
    return [(i, sheets[str(i)]) for i in r.order if str(i) in sheets]


def _unmeasured(r, check, what, floors, required):
    """One item per reason, because a scan gives the same reason on every floor. `floors` is (sheet, reason) pairs."""
    reasons = defaultdict(list)
    for i, why in floors:
        reasons[why].append(i)
    for why, sheets in reasons.items():
        one, names = len(sheets) == 1, ", ".join(r.floor(i) for i in sheets)
        r.add(check, f"{what} on {names if one else f'{len(sheets)} floors'} could not be measured", sheet=sheets[0] if one else None, severity="unknown", measured=why if one else f"{why} Floors: {names}.",
              required=required, evidence=VECTOR, thing=f"{what} {why}")


def _one_exit_rows(r):
    """{sheet: the single exit row with the shortest travel limit, its value made that limit} for each floor that may use one exit: the level of exit discharge
    wherever the rows allow it, because its exits are doors, which are never counted, and each floor above that check 11 finds using its one found stair because the rows allow it."""
    if r.discharge is None:
        return {}
    rows = [row for row, *_ in _established(r)]
    table = _lookup("ibc.1006.3.3", r.jurisdiction, occupant_load=r.load) if rows else None
    floors = [(r.discharge, _single_exit(r, 1))]
    if table:
        floors += [(i, allowed) for i, _, _, allowed in _short_floors(r, rows, table)]
    out = {}
    for i, allowed in floors:
        travel = [row for row in allowed or () if isinstance(row["value"], dict)]   # only Table 1006.3.4(1) rows carry a travel limit
        if travel:
            row = min(travel, key=lambda row: row["value"]["max_exit_access_travel"])
            out[i] = row | {"value": row["value"]["max_exit_access_travel"]}
    return out


def _travel(r):
    floors = _floors_measured(r)
    routed = any(entry.get("routes") for _, entry in floors)   # only a route compared uses a sprinkler value
    limit = _strictest([r.sprinkler_row("ibc.1017.2", occupancy=g) for g in r.groups], min) if routed else None
    one_exit = _one_exit_rows(r) if limit and limit["value"] != "NP" else {}
    _unmeasured(r, 13, "Travel distance", [(i, entry.get("skipped") or "No route was traced.") for i, entry in floors if not entry.get("routes")], "A vector plan with route ends")
    for i, entry in floors:
        single = one_exit[i] if i in one_exit and one_exit[i]["value"] < limit["value"] else None   # the stricter limit wins
        here = limit if single is None or i == r.discharge else single
        for route in entry.get("routes") or ():
            length = route["length_ft"]
            xs, ys = zip(*route["path"])
            where = {"sheet": i, "box": [min(xs), min(ys), max(xs), max(ys)], "measured": f"at least {length:g} ft along the drawn route", "evidence": ROUTE_EVIDENCE, "thing": "route"}
            title = f"Travel distance from {route.get('from_room')} to {route.get('to_exit')}"
            if here is None:
                r.add(13, title, severity="unknown", required="A declared occupancy, which sets the limit", **where)
            elif here["value"] == "NP":
                r.add(13, title, required=_required("Exit access travel is not permitted at this occupancy and sprinkler status.", here), section=_section(here), **where)
            elif length >= ROUTE_MARGIN * here["value"]:
                r.add(13, title, required=_required(f"{here['value']} ft limit", here), section=_section(here), **where)
            elif single and length >= ROUTE_MARGIN * single["value"]:   # the level of exit discharge
                r.add(13, title, severity="unknown", required=_required(f"{single['value']} ft limit if this story has one exit. Its exits are doors, which are not counted.", single),
                      section=_section(single), **(where | {"thing": "one exit route"}))


def _corridors(r):
    minimum = _strictest([_lookup("ibc.1020.3", r.jurisdiction, occupancy=g, occupant_load=r.load) for g in r.groups], max)
    floors = _floors_measured(r)
    _unmeasured(r, 14, "Corridor width", [(i, ("No Hall or Corridor label was read on this floor." if entry.get("scale_in_per_pt") else entry.get("skipped") or "No scale was checked."))
                                          for i, entry in floors if entry.get("scale_in_per_pt") is None or not entry.get("corridors")], "A vector plan with a checked scale and a Hall or Corridor label")
    for i, entry in floors:
        if entry.get("scale_in_per_pt") is None:
            continue
        for hall in entry.get("corridors") or ():
            clear, printed, point = hall.get("clear_in"), hall.get("printed_in"), hall.get("point")
            title = f"Corridor on {r.floor(i)}"
            where = {"sheet": i, "box": _around(point, STAIR_HALF), "thing": "corridor", "place": _cell(point), "evidence": f"{VECTOR} Printed width {printed:g} in, wall centre to wall centre." if printed else VECTOR}
            if clear is None:
                r.add(14, title, severity="unknown", measured="The walls here gave no width.", required="A clear width", **where)
            elif minimum and clear < minimum["value"]:
                r.add(14, title, measured=f"{clear:g} in clear", required=_required(f"{minimum['value']} in", minimum), section=_section(minimum), **where)
            elif printed and printed - clear > WALL_IN:
                r.add(14, f"{title} is narrower than its printed width", measured=f"{clear:g} in clear", required=f"{printed:g} in printed", **where)
            elif minimum is None:
                r.add(14, title, severity="unknown", measured=f"{clear:g} in clear", required="A declared occupancy, which sets the minimum", **where)


def _room_cell(cells):
    """The room name, under ROOM on some sets and NAME on others."""
    return next((v for k, v in cells.items() if "ROOM" in k or "NAME" in k), None)


def _door_room(mark):
    """Door 1A15A opens from room 1A15."""
    return mark[:-1] if re.search(r"\d[A-Z]$", mark) else mark


def _rooms_near(r, s, rooms):
    """The labels a schedule answers to: sheets of its own letter, because modules reuse room numbers, or every sheet if its letter has none."""
    letter = r.number(s["sheet"])[:1]
    return [x for x in rooms if r.number(x["sheet"])[:1] == letter] or rooms


def _numbered(rooms, sheet):
    """Labels by room number, those on `sheet` first."""
    out = {}
    for x in sorted((x for x in rooms if x.get("number")), key=lambda x: x["sheet"] != sheet):
        out.setdefault(_mark(x["number"]), x)
    return out


def _room_names(r):
    rooms = r.declared.get("rooms") or []
    if not rooms:
        return   # declared notes a finish schedule with no room label to compare
    schedules = r.declared.get("schedules") or []
    differ = set()

    def different(got, name, where):
        differ.add((got["sheet"], _mark(got["number"])))
        r.add(16, f"Room {got['number']} name", sheet=got["sheet"], box=got.get("box"), measured=f"{got['name']} on the plan, {name} in the {where} schedule", required=ONE_NAME, evidence=_source(got), thing=f"name {_mark(got['number'])}")

    for s in (s for s in schedules if s["kind"] == "finish"):
        near = _rooms_near(r, s, rooms)
        labels = _numbered(near, s["sheet"])
        printed = {norm(x["name"]) for x in near if not x.get("number")}
        absent = []
        for row in s["rows"]:
            name, got = _room_cell(row["cells"]), labels.get(_mark(row["mark"]))
            if not name or NOT_USED.search(name):
                continue
            if got:
                if not room_same(name, got["name"]):
                    different(got, name, "finish")
            elif norm(name) not in printed:
                absent.append(row)
        if absent:
            unsure = any(x.get("source") == "spark" for x in near) or r.partial(("Finish schedule", "Room labels", "Plan not read"), {s["sheet"], *(x["sheet"] for x in near)})
            r.add(16, f"Rooms in the finish schedule with no label on the plan: {_marks(row['mark'] for row in absent)}", sheet=s["sheet"], box=_union(row.get("box") for row in absent), severity="unknown" if unsure else None,
                  measured=f"{len(absent)} of {len(s['rows'])} rows", required="A plan label for each room in the schedule", evidence=_source(s), thing="no label")
    for s in (s for s in schedules if s["kind"] == "door"):
        labels = _numbered(_rooms_near(r, s, rooms), s["sheet"])
        for row in s["rows"]:
            name, mark = row["cells"].get("ROOM"), _mark(row["mark"])
            got = labels.get(mark) or labels.get(_door_room(mark))
            if name and got and (got["sheet"], _mark(got["number"])) not in differ and not room_same(name, got["name"]):
                different(got, name, "door")


def _feet(inches):
    """Inches as drawings print them. 572.5 gives 47'-8 1/2"."""
    feet, rest = divmod(Fraction(inches).limit_denominator(64), 12)
    whole, part = divmod(rest, 1)
    return f"{feet}'-{whole} {part}\"" if part else f"{feet}'-{whole}\""


def _dimension_strings(r):
    for i, entry in (r.vector.get("sheets") or {}).items():
        for s in entry.get("strings") or ():
            parts = " + ".join(s["parts"]) if len(s["parts"]) <= 10 else some(s["parts"])
            r.add(17, f"Dimension string does not add up to {s['overall']}", sheet=int(i), box=s["box"], measured=f"{parts} = {_feet(s['sum_in'])}, and the overall says {s['overall']}.", required="A dimension string adds up to its overall",
                  evidence=VECTOR, thing=f"string {s['parts']} {s['overall']}")


def _read_notes(r):
    """Every cannot_tell from declared stays visible."""
    for n in r.notes:
        what = n["what"]
        r.add(4 if what.startswith("Code field") else NOTE_CHECKS.get(what, 1), f"{what}: {n['why']}", sheet=n.get("sheet"), severity="unknown", evidence="Declared reads.", thing=what)


# Label for not_checked, the check, and the stages it reads.
CHECKS = [
    ("required sheets", _required_sheets, ("read", "declared")),
    ("sheets not read", _unread_sheets, ("read",)),
    ("plans outside the floor order", _plans_left_out, ("read", "floors")),
    ("sheet index", _sheet_index, ("declared",)),
    ("sheet titles", _titles, ("declared",)),
    ("code summary", _code_summary, ("declared",)),
    ("stories against floor plans", _stories, ("read", "floors", "declared")),
    ("code edition", _edition, ("declared",)),
    ("table arithmetic", _tables, ("declared",)),
    ("door and window tags", _tags, ("declared",)),
    ("door type against its swing", _swings, ("declared", "vector")),
    ("window sizes", _window_types, ("declared",)),
    ("swing column", _swing_materials, ("declared",)),
    ("exit capacity", _exit_capacity, ("declared",)),
    ("stair continuity", _stair_gaps, ("read", "floors", "verticals")),
    ("stairs at the discharge level", _stair_discharge, ("read", "floors", "verticals")),
    ("exits per floor", _exits_per_floor, ("read", "floors", "verticals", "declared")),
    ("stair list", _stair_list, ("read", "floors", "verticals")),
    ("travel distance", _travel, ("read", "floors", "vector", "declared")),
    ("corridor width", _corridors, ("read", "floors", "vector", "declared")),
    ("room names", _room_names, ("declared",)),
    ("dimension strings", _dimension_strings, ("vector",)),
    ("reads that did not finish", _read_notes, ("declared",)),
]
RULE_BASED = ("code edition", "exits per floor", "travel distance", "corridor width")   # moved to not_checked on a dwelling, as is the life safety sheet within required sheets. No adopted residential code is held, so an edition has nothing to compare with


def _unique(items):
    """Two stairs can share a grid cell, and a key must name one item."""
    seen = Counter()
    for item in items:
        seen[item["key"]] += 1
        if seen[item["key"]] > 1:
            item["key"] += f"#{seen[item['key']]}"


def _masked(value):
    """A value with status words taken out. Drawing text reaches the output in many fields, a planted room label can say the set is approved, and the letter prints what it is given."""
    if isinstance(value, str):
        text = STATUS.sub(MASK, value)
        return MASK if has_status(text) else text   # the whole value goes, because normalising it in place would turn ft\u00b2 into ft2
    if isinstance(value, list):
        return [_masked(v) for v in value]
    if isinstance(value, dict):
        return {k: _masked(v) for k, v in value.items()}
    return value


def evaluate(stages: dict, jurisdiction: str, sprinklers: str | None, discharge: int | None) -> dict:
    """{"findings": [...], "open_items": [...], "not_checked": [...]}. `sprinklers` ("yes" or "no") and `discharge` (a sheet index) are the examiner's choices, and None leaves the code sheet's statement and the floors stage's label standing.
    Stage results are read as stored, so maps keyed by sheet index have string keys. Open items have key, check, sheet, box, title, evidence and vertical. Raises KeyError for an unknown jurisdiction."""
    JURISDICTIONS[jurisdiction]   # a typo here would otherwise read as rule rows with no case
    lost = {name: (stages.get(name) or {}).get("failed", "It has no result.") for name in STAGES if not stages.get(name) or "failed" in stages[name]}
    r = Review(**{name: {} if name in lost else stages[name] for name in STAGES}, jurisdiction=jurisdiction, sprinklers=sprinklers, discharge=discharge)
    skipped, broken = defaultdict(list), []
    for label, check, needs in CHECKS:
        down = [name for name in needs if name in lost]
        for name in down:
            skipped[name].append(label)
        if down or (r.dwelling and label in RULE_BASED):
            continue
        try:
            check(r)
        except Exception as e:   # stage text is hostile and stage shapes are not proven, and one check must not cost the others
            log.warning("The %s check failed with %s", label, type(e).__name__)
            broken.append(f"The {label} check did not finish.")
    if r.stricter:
        r.add(4, "Sprinkler status is not known, so the stricter value was used", severity="unknown", measured="No selection and no sprinkler statement on the code sheet.", required="Examiner to confirm the sprinkler status", evidence="Rule rows.", thing="stricter")
    r.findings.sort(key=lambda f: RANK[f["severity"]])
    _unique(r.findings)
    _unique(r.open_items)
    # shortcut: this is the floors rule, so beside an A sheet a plan sheet such as B2.0, X4.1 or one with no number gets no line, and with no A sheet a trade sheet still gets one, share assemble's trade rule if a real set shows either.
    lines = [*(f"The {name} stage did not finish: {lost[name]} Not checked because of it: {', '.join(labels)}." for name, labels in skipped.items()), *broken, r.floors.get("not_checked"), *(r.verticals.get("not_checked") or ()),
             *(SEVERAL_PLANS.format(r.number(s["index"])) for s in _plan_sheets(list(r.sheets.values())) if len(s.get("viewports") or ()) >= 2),   # one box or none says nothing, because SAM misses plans
             SHAFTS, OTHER_TRADES, DWELLING if r.dwelling else None]
    return _masked({"findings": r.findings, "open_items": r.open_items, "not_checked": list(dict.fromkeys(line for line in lines if line))})


def _drift_pairs(new, free):
    """(distance, new key, old key) for each new key and unclaimed old key of one stair finding at most one grid cell apart, nearest first. Finding order follows row id, which changes between fresh runs, so it must not decide who claims a key."""
    # shortcut: a key holds only its cell, so pairs at one distance go by key order, compare sheet positions if a real review swaps two stairs.
    pairs = []
    for key in new:
        m = PLACED.fullmatch(key)
        for old in free if m else ():
            o = PLACED.fullmatch(old)
            if o and o[1] == m[1]:
                dx, dy = int(o[2]) - int(m[2]), int(o[3]) - int(m[3])
                if abs(dx) <= 1 and abs(dy) <= 1:
                    pairs.append((dx * dx + dy * dy, key, old))
    return sorted(pairs)


def assign_ids(findings: list[dict], previous: dict[str, str]) -> None:
    """Sets `comment_id`: the id `previous` holds for the key, else the next number after any used. A stair that drifted across a grid line takes its old key too, so the comment carries on."""
    free, new = dict(previous), {}
    for f in findings:
        if f["key"] in free:
            f["comment_id"] = free.pop(f["key"])
        else:
            new[f["key"]] = f
    for _, key, old in _drift_pairs(new, free):
        if key in new and old in free:
            f = new.pop(key)
            f["key"], f["comment_id"] = old, free.pop(old)
    used = max((int(m[0]) for v in previous.values() if (m := re.search(r"\d+$", v))), default=0)
    for n, f in enumerate(new.values(), used + 1):
        f["comment_id"] = f"{DISCIPLINE}-{n:02d}"
