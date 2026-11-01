"""Planted stair erasures on copies of real sheets, so the stair rule's wrong passes can be counted. One copy holds several erasures on different floors, so one run covers them all.

The cells come from a truth file read by eye before any model call, never from Storey's rows, so a stair Storey did not find can be planted too.
The removal is the test tool's: the stair goes and its walls stay, as on the plants the rule was tested on.

    python -m eval.plant habs_stelizabeths_center

writes data/eval/plants/<building>_plant/: the sheets in floor order as 000.png, 001.png and on, truth.json, and check.jpg, each erasure before and after.
truth.json also holds every stair of the building with its box on each floor where it is drawn, so a stair shown Found on a floor where it is not drawn counts as a wrong pass too.
Look at check.jpg before a run: a truth box drawn tighter than its stair leaves treads outside the box, and a Found there would be the plant's fault, not Storey's.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from storey.floors import WORK_PX, ink
from storey.pipeline import write_json
from storey.testtool import _remove

ROOT = Path(__file__).resolve().parent.parent
TRUTH = ROOT / "lab/presence/truth.json"   # boxes in px of SHEETS/<building>/<floor>.jpg, with a plant list per building fixed before any model output
SHEETS = ROOT / "data/lab/presence/sheets"
OUT = ROOT / "data/eval/plants"
EXPECTED = {"middle": 9, "discharge": 10, "top": None}   # the check that should raise each kind. None on a stair's top floor: an erasure there reads like a stair that ends, the stated limit
PAD = 40   # px of sheet round each box in check.jpg


def _ink(img, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    return round(float(ink(img)[y1:y2, x1:x2].mean()), 4)


def _cut(img, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    return cv2.resize(img[max(y1 - PAD, 0):y2 + PAD, max(x1 - PAD, 0):x2 + PAD], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


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
        path = folder / names[index]
        before = cv2.imread(str(path))
        _remove(path, box)
        after = cv2.imread(str(path))
        rows.append({"object": name, "floor": floor, "index": index, "kind": kind, "expected": EXPECTED[kind], "box": [round(v * scale[index], 1) for v in box],
                     "ink_before": _ink(before, box), "ink_after": _ink(after, box)})
        old, new = _cut(before, box), _cut(after, box)
        tiles.append(np.hstack([old, np.zeros((old.shape[0], 8, 3), np.uint8), new]))
    width = max(t.shape[1] for t in tiles)
    cv2.imwrite(str(folder / "check.jpg"), np.vstack([cv2.copyMakeBorder(t, 4, 4, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    stairs = {name: {str(order.index(int(f))): [round(v * scale[order.index(int(f))], 1) for v in box] for f, box in o["boxes"].items() if int(f) in order}
              for name, o in objects.items() if o["kind"] == "stair"}   # keyed by sheet index, like a review's cells
    write_json(folder / "truth.json", {"set": folder.name, "source": building, "from": str(truth), "from_sha256": hashlib.sha256(Path(truth).read_bytes()).hexdigest(),
                                       "box_units": "px of the review's 2000 px sheet", "sheets": names, "floors": order, "stairs": stairs, "cells": rows})
    return folder


if __name__ == "__main__":
    print(plant(sys.argv[1]))
