"""The evaluate stage, and review.json, the one file the screen reads. Stages store facts in sheet pixels and the screen draws on 2000 px JPEGs, so every point, box and mask converts here."""
import functools
import re
from dataclasses import replace
from pathlib import Path

import cv2
from PIL import Image

from .findings import STAGES as FACT_STAGES, Review, _left_out, _masked, assign_ids, evaluate
from .floors import WORK_PX, _resized
from .letter import draft
from .rules import JURISDICTIONS

JURISDICTION = "houston"
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 92]
CROP_PAD = 100     # sheet px round a finding's box, about a door and its tag on a 6000 px sheet
CROP_MAX = 1600    # px, because a finding about a whole plan unions tags across the sheet
TRADE_NUMBER = re.compile(r"(?i)^[EPMS][-.\d]")   # the discipline letter of the US CAD standard, where A is architectural
TRADE_TITLE = re.compile(r"(?i)electrical|plumbing|mechanical|structural|zoning|energy")
NOT_READ = "This sheet could not be read."
OTHER_TRADE = "Not an architectural sheet, so only its number and title were read."
LEFT_OUT = "This plan is outside the floor order, so its stairs, travel distance and corridor width were not checked."
ALL_DROPPED = "The examiner dropped every comment, so this letter lists none."


def _evaluated(stored, choices, previous):
    """findings.evaluate's result with comment ids, and what it ran under. `ids` keeps every id issued, so a finding that leaves under one jurisdiction takes its old id when it comes back."""
    result = evaluate({name: stored(name) for name in FACT_STAGES}, **choices)
    assign_ids(result["findings"], previous)
    return result | {"choices": choices, "ids": previous | {f["key"]: f["comment_id"] for f in result["findings"]}}


def evaluate_stage(ctx) -> dict:
    """Comment ids go on here, because the letter stage reads them. A second review starts from its parent's ids, jurisdiction and sprinklers, so a comment keeps its id across cycles and the letter cites the same code."""
    parent = ctx.stage("upload").get("parent")
    earlier = replace(ctx, dir=ctx.dir.parent / parent).stage("evaluate") if parent else {}
    chosen = earlier.get("choices", {})
    choices = {"jurisdiction": chosen.get("jurisdiction", JURISDICTION), "sprinklers": chosen.get("sprinklers"), "discharge": None}   # discharge is a sheet index, which does not carry across sets
    return _evaluated(ctx.stage, choices, earlier.get("ids", {}) | ctx.stage("evaluate").get("ids", {}))


def _image(dir, sheet):
    """(file name, width, height) of the sheet's JPEG. It is made once, because a Keep click must not decode every sheet again."""
    path = dir / f"{Path(sheet['image']).stem}.jpg"
    if not path.exists():
        cv2.imwrite(str(path), _resized(cv2.imread(str(dir / sheet["image"])), WORK_PX), JPEG)
    with Image.open(path) as image:
        return path.name, *image.size


def _crop(dir, f, load):
    """The file name of a finding's crop, cut once from the full sheet because the 2000 px copy is too coarse to read a tag, or None with no place to cut."""
    if f["sheet"] is None or not f["box"]:
        return None
    name = f"crop_{f['comment_id']}.jpg"
    if not (dir / name).exists():
        x1, y1, x2, y2 = (int(v) for v in f["box"])
        cut = load(f["sheet"])[max(y1 - CROP_PAD, 0):max(y2 + CROP_PAD, 0), max(x1 - CROP_PAD, 0):max(x2 + CROP_PAD, 0)]
        if not cut.size:
            return None
        cv2.imwrite(str(dir / name), _resized(cut, CROP_MAX) if max(cut.shape[:2]) > CROP_MAX else cut, JPEG)
    return name


def _scaled(values, k):
    """A flat [x, y, ...] from sheet pixels to that sheet's JPEG pixels. None stays None."""
    return values and [round(v * k[n % 2], 1) for n, v in enumerate(values)]


def _placed(item, scales):
    """An item with its sheet as the screen's id and its box in that sheet's JPEG. A box with no sheet places nothing."""
    sheet = item.get("sheet")
    return item | {"box": None} if sheet is None else item | {"sheet": str(sheet), "box": _scaled(item.get("box"), scales[sheet])}


def _coverage(row, lost, left_out):
    if not row.get("read_ok"):
        return False, NOT_READ
    if TRADE_NUMBER.match(row["number"]) or TRADE_TITLE.search(row["title"]):
        return False, OTHER_TRADE
    if lost:
        return False, f"Stages that did not finish: {', '.join(lost)}. This sheet was not fully checked."
    if row["index"] in left_out:
        return False, LEFT_OUT
    return True, None


def _sheet(sheet, row, image, url, lost, left_out):
    i, (name, w, h) = sheet["index"], image
    reviewed, reason = _coverage(row, lost, left_out)
    return {"id": str(i), "number": row.get("number", ""), "level": row.get("level") or row.get("number") or f"Page {i + 1}", "title": row.get("title", ""), "image": url(name), "w": w, "h": h,
            "class": row.get("cls"), "kind": sheet["kind"], "scale": row.get("scale", ""), "reviewed": reviewed, "reason": reason}


def _cell(cell, k, url):
    return cell | {"point": _scaled(cell["point"], k), "box": _scaled(cell["box"], k), "mask": cell["mask"] and [_scaled(p, k) for p in cell["mask"]], "crop": cell["crop"] and url(cell["crop"])}


def _vertical(row, scales, reference, url):
    return row | {"pos": row["pos"] and _scaled(row["pos"], scales[reference]), "floors": {i: _cell(cell, scales[int(i)], url) for i, cell in row["floors"].items()}}


def _routes(vector, scales):
    return [{"sheet": i, "path": [_scaled(p, scales[int(i)]) for p in route["path"]], "length": route["length_ft"], "label": route["label"]}
            for i, entry in (vector.get("sheets") or {}).items() for route in entry.get("routes") or ()]


def _travel_checked(evaluated):
    """A route drawn where travel distance was not checked would look measured."""
    # shortcut: matches the words of evaluate's lines, each of which names the check, read a list of skipped checks from evaluate if those lines are reworded.
    return "findings" in evaluated and not any("travel distance" in line for line in evaluated["not_checked"])


def _lost(stored, evaluated):
    """Lines for stages evaluate could not report: the letter, and if evaluate itself did not finish, every stage before it too."""
    names = ("letter",) if "findings" in evaluated else (*FACT_STAGES, "evaluate", "letter")
    return [f"The {name} stage did not finish: {stored(name)['failed']}" for name in names if "failed" in stored(name)]


def assemble(dir: Path, jurisdiction: str = JURISDICTION, sprinklers: str | None = None, discharge: int | None = None) -> dict:
    """Writes review.json from the stored stages and returns it. Choices that differ from the stored ones run the checks again, and any choices but the first run's use a letter rebuilt from its template, so no model is called.
    stage_letter.json stays as the stage wrote it, so the model's wording returns with the first run's choices. An evaluate stage that did not finish has no stored choices, and then this reports the failure rather than run it again."""
    from .pipeline import read_json, write_json   # here, because pipeline imports this module to register the stages

    def stored(name):
        return read_json(dir / f"stage_{name}.json")

    def url(name):
        return f"/api/reviews/{dir.name}/files/{name}"

    choices = {"jurisdiction": jurisdiction, "sprinklers": sprinklers, "discharge": discharge}
    evaluated, letter = stored("evaluate"), stored("letter")
    first = evaluated.get("first", evaluated.get("choices", choices))   # what the letter stage ran under, kept below when a selection replaces this file
    if evaluated.get("choices", choices) != choices:
        evaluated = _evaluated(stored, choices, evaluated["ids"]) | {"first": first}
        write_json(dir / "stage_evaluate.json", evaluated)
    if first != choices:
        letter = draft(evaluated["findings"], {"sheets": stored("read").get("sheets", []), "not_checked": evaluated["not_checked"]})
    sheets = {s["index"]: s for s in read_json(dir / "sheets.json") or ()}
    images = {i: _image(dir, s) for i, s in sheets.items()}
    scales = {i: (w / sheets[i]["width"], h / sheets[i]["height"]) for i, (_, w, h) in images.items()}
    rows = {r["index"]: r for r in stored("read").get("sheets", ())}
    floors, upload, costs = stored("floors"), stored("upload"), read_json(dir / "cost.json")
    order = floors.get("order") or []
    level = floors.get("discharge") if discharge is None else discharge
    parent = upload.get("parent")
    earlier = read_json(dir.parent / parent / "review.json") if parent else {}
    decisions = earlier.get("decisions", {}) | read_json(dir / "decisions.json")   # review.json, so a decision from any earlier cycle arrives
    kept = [item for item in letter.get("items", ()) if decisions.get(item["key"]) != "drop"]
    lost = [name for name in (*FACT_STAGES, "evaluate") if not (stage := stored(name)) or "failed" in stage]
    left_out = {s["index"] for s in _left_out(Review(stored("read"), floors, {}, {}, {}, jurisdiction, None, None))}   # findings' own test, so a row and its finding agree
    adopted = JURISDICTIONS[jurisdiction]
    load = functools.lru_cache(maxsize=4)(lambda i: cv2.imread(str(dir / sheets[i]["image"])))   # one sheet can hold many findings, and a 6000 px sheet takes a moment to open
    facts = {name: _placed({k: row.get(k) for k in ("value", "sheet", "box", "reads")}, scales) for name, row in (stored("declared").get("code") or {}).items()}
    review = {
        "id": dir.name,
        "parent": parent,
        "project": {"name": upload.get("name", ""), "address": "", "cycle": earlier.get("project", {}).get("cycle", 1) + 1 if parent else 1,
                    "code": adopted, "note": "", "jurisdiction": jurisdiction, "adopted_code": adopted, "sprinklers": sprinklers, "discharge_level": str(level) if level in order else None, "facts": facts},
        "floor_order": [str(i) for i in order],
        "sheets": [_sheet(s, rows.get(i, {}), images[i], url, lost, left_out) for i, s in sheets.items()],
        "verticals": [_vertical(row, scales, floors.get("reference"), url) for row in stored("verticals").get("verticals", ())],
        "findings": [_placed(f, scales) | {"id": f["comment_id"], "state": None, "crop": (name := _crop(dir, f, load)) and url(name)} for f in evaluated.get("findings", ())],
        "routes": _routes(stored("vector"), scales) if _travel_checked(evaluated) else [],
        "open_items": [_placed(item, scales) for item in evaluated.get("open_items", ())],
        "not_checked": [*evaluated.get("not_checked", ()), *_lost(stored, evaluated)],
        "letter": letter | {"items": kept, "note": ALL_DROPPED if letter.get("items") and not kept else letter.get("note")},
        "decisions": decisions,
        # shortcut: this stage's own row reaches cost.json only after this file is written, so review.json has no assemble row, rewrite the file after the run if the screen must show it.
        "cost": {"usd": round(sum(c["usd"] for c in costs.values()), 4), "seconds": round(sum(c["seconds"] for c in costs.values()), 1), "stages": [{"name": name, **c} for name, c in costs.items()]},
    }
    review = _masked(review)   # sheet titles and upload names reach this file with no check of their own
    write_json(dir / "review.json", review)
    return review


def assemble_stage(ctx) -> dict:
    assemble(ctx.dir, **ctx.stage("evaluate").get("choices", {}))
    return {"review": "review.json"}
