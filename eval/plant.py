"""Planted stair erasures on copies of real sheets, so the stair rule's wrong passes can be counted. One copy holds several erasures on different floors, so one run covers them all.

The cells come from a truth file read by eye before any model call, never from Storey's rows, so a stair Storey did not find can be planted too.
The removal is the test tool's: the stair goes and its walls stay, as on the plants the rule was tested on.

    python -m eval.plant habs_stelizabeths_center
    python -m eval.plant army_bt_bcof

writes data/eval/plants/<building>_plant/: the sheets in floor order as 000.png, 001.png and on, truth.json, and check.jpg, each erasure before and after.
truth.json also holds every stair of the building with its box on each floor where it is drawn, so a stair shown Found on a floor where it is not drawn counts as a wrong pass too.
A held-out set, one with a stair truth in eval/truth, is planted from that truth alone: every page the set uploads, drawn as ingest draws a PDF page, with the cells `erasures` picks taken off.
Its truth.json holds the stair truth placed on the review's 2000 px sheets by score.placed, with the boxes where truth says a stair is absent.
No held-out plant is written when an erasure leaves its treads: thin ink in the box, the strokes the removal takes, must fall below INK_LOW, the stair rule's own line, of what it was.
The removal was made for black lines on paper. On queens_royce's colour-filled raster it left a flight of MIL_NW and put white patches over the other two cells, so that set has no plant.
Each truth.json written here records each image's sha256, and eval.run refuses an image changed after the plant was made.
Look at check.jpg before a run: a truth box drawn tighter than its stair leaves treads outside the box, and a Found there would be the plant's fault, not Storey's.
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from storey.floors import WORK_PX, _resized, ink
from storey.ingest import _render
from storey.pipeline import write_json
from storey.testtool import THICK, _remove
from storey.verticals import INK_LOW

from . import run, score

ROOT = Path(__file__).resolve().parent.parent
TRUTH = ROOT / "lab/presence/truth.json"   # boxes in px of SHEETS/<building>/<floor>.jpg, with a plant list per building fixed before any model output
SHEETS = ROOT / "data/lab/presence/sheets"
OUT = run.PLANTS
EXPECTED = {"middle": 9, "discharge": 10, "top": None}   # the check that should raise each kind. None on a stair's top floor: an erasure there reads like a stair that ends, the stated limit
PAD = 40   # px of sheet round each box in check.jpg


def _ink(img, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    return round(float(ink(img)[y1:y2, x1:x2].mean()), 4)


def _cut(img, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    return cv2.resize(img[max(y1 - PAD, 0):y2 + PAD, max(x1 - PAD, 0):x2 + PAD], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def _erased(path, box):
    """Takes the stair in `box`, in the image's own px, off the sheet image at `path`: (ink in the box before, after, the before and after tile for check.jpg)."""
    before = cv2.imread(str(path))
    _remove(path, box)
    after = cv2.imread(str(path))
    old, new = _cut(before, box), _cut(after, box)
    return _ink(before, box), _ink(after, box), np.hstack([old, np.zeros((old.shape[0], 8, 3), np.uint8), new])


def _check(folder, tiles):
    width = max(t.shape[1] for t in tiles)
    cv2.imwrite(str(folder / "check.jpg"), np.vstack([cv2.copyMakeBorder(t, 4, 4, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def plant(building, truth=TRUTH, sheets=SHEETS, out=OUT):
    """Writes the planted copy of one building and returns its folder."""
    data = json.loads(Path(truth).read_text())
    objects, cells = data["buildings"][building]["objects"], data["plant"][building]
    floors = [floor for _, _, floor in cells]
    if len(set(floors)) != len(floors):
        raise ValueError("Plant one stair per floor, so the evidence on each floor answers for one erasure.")
    sources = sorted((p for p in (Path(sheets) / building).glob("*.jpg") if p.stem.isdigit()), key=lambda p: int(p.stem))
    folder = Path(out) / f"{building}_plant"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    names = [f"{int(p.stem):03d}.png" for p in sources]   # ingest takes page images in name order
    order, scale = [int(p.stem) for p in sources], []
    for p, name in zip(sources, names):
        image = cv2.imread(str(p))
        cv2.imwrite(str(folder / name), image)
        scale.append(WORK_PX / max(image.shape[:2]))   # the review draws on a 2000 px copy
    rows, tiles = [], []
    for kind, name, floor in cells:
        box = objects[name]["boxes"].get(str(floor))
        if box is None:
            raise ValueError(f"{name} is not drawn on floor {floor}, so there is nothing to erase.")
        index = order.index(floor)
        before, after, tile = _erased(folder / names[index], box)
        rows.append({"object": name, "floor": floor, "index": index, "kind": kind, "expected": EXPECTED[kind], "box": [round(v * scale[index], 1) for v in box], "ink_before": before, "ink_after": after})
        tiles.append(tile)
    _check(folder, tiles)
    stairs = {name: {str(order.index(int(f))): [round(v * scale[order.index(int(f))], 1) for v in box] for f, box in o["boxes"].items() if int(f) in order}
              for name, o in objects.items() if o["kind"] == "stair"}   # keyed by sheet index, like a review's cells
    write_json(folder / "truth.json", {"set": folder.name, "source": building, "from": str(truth), "from_sha256": hashlib.sha256(Path(truth).read_bytes()).hexdigest(), "png_sha256": {n: run._sha(folder / n) for n in names},
                                       "box_units": "px of the review's 2000 px sheet", "sheets": names, "floors": order, "stairs": stairs, "cells": rows})
    return folder


def erasures(stairs):
    """[(kind, stair id, floor name)] a held-out plant erases, picked from its stair truth alone. Floors go lowest first, as each stair truth lists them, and the top floor gets none.
    Each other floor takes the first stair in id order whose cell there is present with confidence certain, that is present on the floor above, so the erasure is not on its top floor,
    and that no lower floor took, so no stair loses two floors. A floor with no such stair takes none. The lowest floor's erasure is a discharge plant and the others are middle ones."""
    cells = {(s["id"], c["page"]): (c["state"], c["confidence"]) for s in stairs["stairs"] for c in s["cells"]}
    ids, picked = sorted(s["id"] for s in stairs["stairs"]), []
    for k, (floor, above) in enumerate(zip(stairs["floors"], stairs["floors"][1:])):
        taken = [stair for _, stair, _ in picked]
        stair = next((s for s in ids if s not in taken and cells.get((s, floor["page"])) == ("present", "certain") and cells.get((s, above["page"]), (None, None))[0] == "present"), None)
        if stair:
            picked.append(("discharge" if k == 0 else "middle", stair, floor["name"]))
    return picked


def _thin(path, box):
    """Ink in `box` thinner than a wall, in px of the 2000 px copy the removal works on: the treads it takes, without the walls it keeps."""
    marks = np.uint8(ink(_resized(cv2.imread(str(path)), WORK_PX)) > 0)
    walls = cv2.morphologyEx(marks, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (THICK, THICK)))
    x1, y1, x2, y2 = (int(v) for v in box)
    return round(float((marks > walls)[y1:y2, x1:x2].mean()), 4)


def held_out(name, out=OUT):
    """Writes the planted copy of a held-out set and returns its folder. When an erasure leaves its treads, it raises ValueError and leaves no folder."""
    source = run.TRUTH / f"{name}_stairs.json"
    stairs = json.loads(source.read_text())
    (pdf, numbers), = run.inputs(name)
    folder = Path(out) / f"{name}_plant"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    try:
        # shortcut: page images reach ingest as scans with no text or vector layer, where an upload of the PDF has both, plant into the PDF itself if a check that reads either must be scored on a plant.
        with tempfile.TemporaryDirectory() as tmp:
            run._upload(pdf, numbers, Path(tmp) / "0000.pdf")   # the file a run of the set uploads: queens_royce page 9 renders a little differently from its source
            with pymupdf.open(Path(tmp) / "0000.pdf") as doc:
                names = [f"{i:03d}.png" for i in range(doc.page_count)]   # ingest takes page images in name order
                sizes = [_render(page, folder / n)[:2] for page, n in zip(doc, names)]   # ingest's own render of a PDF page
        full = score.placed(name, stairs, sizes)
        small = score.placed(name, stairs, [(round(w * WORK_PX / max(w, h)), round(h * WORK_PX / max(w, h))) for w, h in sizes])   # the review draws on a 2000 px copy
        rows, tiles = [], []
        for kind, stair, floor in erasures(stairs):
            i = full["floors"].index(floor)
            path, box = folder / names[i], small["stairs"][stair][str(i)]
            thin = _thin(path, box)
            before, after, tile = _erased(path, full["stairs"][stair][str(i)])
            rows.append({"object": stair, "floor": floor, "index": i, "kind": kind, "expected": EXPECTED[kind], "box": box, "ink_before": before, "ink_after": after, "thin_ink_before": thin, "thin_ink_after": _thin(path, box)})
            tiles.append(tile)
        # shortcut: the gate reads only ink thinner than a wall, so it cannot see treads drawn as thick as walls and passes them once the thin ink round them goes, read the treads another way if a set draws them thick.
        left = [r for r in rows if r["thin_ink_after"] >= INK_LOW * r["thin_ink_before"]]   # none before means the ink map sees no treads in the box, so the erasure cannot be checked
        if left:
            raise ValueError(f"No plant is written, because thin ink in the box, the treads the removal takes, was none to begin with or did not fall below {INK_LOW} of what it was, for "
                             + ", ".join(f"{r['object']} on {r['floor']} ({r['thin_ink_before']} before, {r['thin_ink_after']} after)" for r in left) + ".")
        _check(folder, tiles)
        write_json(folder / "truth.json", {"set": folder.name, "source": name, "from": str(source), "from_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "png_sha256": {n: run._sha(folder / n) for n in names},
                                           "box_units": "px of the review's 2000 px sheet", **small, "cells": rows})
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder


if __name__ == "__main__":
    name = sys.argv[1]
    print(held_out(name) if (run.TRUTH / f"{name}_stairs.json").exists() else plant(name))
