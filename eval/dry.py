"""Runs a generated set through the stages that read the PDF, with no model call: what Spark would read is taken from the model.

It shows what the text layer and vector readers make of a drawing before any money is spent, and it is what the tests run. It proves nothing about what Spark reads from the sheets.
"""
import json
from pathlib import Path

import pymupdf

import storey.declared as declared
from storey.findings import STAGES, evaluate
from storey.meta import Meter
from storey.pipeline import Context, write_json
from storey.vector import vector_stage

from . import draw

CLS = {"cover": "cover", "life": "floor_plan", "plan": "floor_plan", "door": "schedule", "window": "schedule", "finish": "schedule"}
PX = 2000 / 2592        # enough to crop from, and a real run's sheets are 6000 px wide with every box scaled the same way


def _rows(m, sheets, k):
    rows = []
    for i, s in enumerate(sheets):
        f = m.floors[s.floor - 1] if s.floor else None
        t = {"plan": draw.plan_frame, "life": draw.life_frame}.get(s.kind)
        box = [round(v * k) for v in t(m).rect(f.outline)] if t else [0, 0, round(2592 * k), round(1728 * k)]
        rows.append({"index": i, "number": s.number, "title": s.block_title or s.title, "cls": CLS[s.kind], "level": f.name if f else "", "scale": {"plan": m.scale[1], "life": m.ls_scale[1]}.get(s.kind, ""), "plan_box": box, "read_ok": True})
    return rows


def _stairs(m, sheets, k):
    plans = {s.floor: i for i, s in enumerate(sheets) if s.kind == "plan"}
    cells = {}
    for f in m.floors:
        x0, y0, x1, y1 = draw.plan_frame(m).rect(draw._stairs(f))
        box = [round(v * k) for v in (x0, y0, x1, y1)]
        state = "found" if f.stair_on_plan else "missing"
        cells[str(plans[f.number])] = {"state": state, "signals": {"spark": "3/3", "crop": "yes", "ink": 1.0, "aligned": True, "reason": "from the model"}, "point": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], "box": box if f.stair_on_plan else None, "crop": None, "mask": None}
    return {"verticals": [{"id": "A", "name": "Stair 1", "established": True, "pos": next(iter(cells.values()))["point"], "floors": cells}], "not_checked": []}


def _exits(m, sheets, k):
    rows = []
    for i, s in enumerate(sheets):
        if s.kind == "life":
            t = draw.life_frame(m)
            marks = [{"point": [round(v * k) for v in t(e["point"])], "width_mm": e["width"] * 25.4, "capacity": e["capacity"]} for e in m.markers[s.floor]]
            total = sum(e["capacity"] for e in marks)
            rows.append({"sheet": i, "sum": total, "reads": [total, total], "markers": marks})
    return {"sheets": rows}


def _spark(m):
    def reply(prompt, images, **_):
        if prompt == declared.CODE_P:
            code = m.code
            return {"code_edition": code["edition"], "occupancy": code["occupancy"], "occupancy_mentions": [], "construction": code["construction"], "sprinklered": "yes", "sprinkler_mentions": [code["sprinklers"]], "stories": code["stories"], "area": code["area"], "occupant_load": code["load"]}
        if prompt == declared.EXIT_P:
            return {"exits": [], "travel_distances": [], "occupant_loads": []}
        if prompt.startswith(declared.JUDGE_P):
            return {"results": [{"id": pair["id"], "answer": "different"} for pair in json.loads(prompt[len(declared.JUDGE_P):])]}
        raise AssertionError(f"An unexpected read in a dry run: {prompt[:60]}")
    return reply


def review(m, pdf, folder):
    """Stage files for one set in `folder`, and {jurisdiction: findings}, with Spark's reads replaced. A set that needs another Spark read fails here."""
    folder = Path(folder)
    (folder / "upload").mkdir(parents=True, exist_ok=True)
    (folder / "upload/0000.pdf").write_bytes(Path(pdf).read_bytes())
    sheets = [s for s in m.sheets if s.in_set]
    rows = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=pymupdf.Matrix(PX, PX), alpha=False)
            pix.save(folder / f"sheet_{i:03d}.png")
            rows.append({"index": i, "image": f"sheet_{i:03d}.png", "width": pix.width, "height": pix.height, "kind": "vector"})
    write_json(folder / "sheets.json", rows)
    ctx = Context(folder, Meter(ledger=folder / "ledger.jsonl"), rows, False, {})
    write_json(folder / "stage_read.json", {"sheets": _rows(m, sheets, PX)})
    plans = [i for i, s in enumerate(sheets) if s.kind == "plan"]
    write_json(folder / "stage_floors.json", {"order": plans, "order_read": True, "reference": plans[0], "transforms": {}, "inliers": {}, "unaligned": [], "stack": None, "frame_size": None, "discharge": plans[0], "not_checked": None})
    write_json(folder / "stage_verticals.json", _stairs(m, sheets, PX))
    live = declared.spark
    declared.spark = _spark(m)
    try:
        stage = declared.declared_stage(ctx)
    finally:
        declared.spark = live
    stage["exits"] = _exits(m, sheets, PX)
    stage["cannot_tell"] = [n for n in stage["cannot_tell"] if n["what"] != "Exit capacity"]
    write_json(folder / "stage_declared.json", stage)
    write_json(folder / "stage_vector.json", vector_stage(ctx))
    stages = {name: json.loads((folder / f"stage_{name}.json").read_text()) for name in STAGES}
    return {city: evaluate(stages, city, None, None) for city in ("houston", "seattle")}
