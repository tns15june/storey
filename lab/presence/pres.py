"""Shared paths and geometry for the presence experiment. Imports lab/align/reg.py unedited and repoints its globals at this experiment's sheets."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "lab"))
sys.path.insert(0, str(ROOT / "lab/align"))
import reg  # noqa: E402

OUT = ROOT / "data/lab/presence"
SH = OUT / "sheets"
INDEX = json.load(open(SH / "index.json"))
TRUTH = json.load(open(ROOT / "lab/presence/truth.json"))
T = TRUTH["buildings"]
DEV = [b for b, t in T.items() if t["split"] == "dev"]
HELD = [b for b, t in T.items() if t["split"] == "heldout"]
GATE = 25  # inliers, from Round 1's negative control (unrelated buildings reach 10)
reg.OUT = OUT
reg.INDEX = INDEX


def load_spark():
    reg.SPARK = json.load(open(OUT / "spark_sheets.json"))
    return reg.SPARK


def nfloors(b):
    return len(INDEX[b])


def ref_of(b):
    return 1 if nfloors(b) > 2 else 0


def size(b, f):
    h, w = INDEX[b][int(f)]["shape"]
    return w, h


def apply_pt(H, p):
    q = H @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


def box_dist(p, b):
    """Pixels from a point to a box, 0 inside."""
    return float(np.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3])))


def ctr(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def alignment(b):
    return json.load(open(OUT / f"align/{b}.json"))


def fam(kind):
    return "stair" if kind in ("stair", "stairs") else "shaft"


if __name__ == "__main__":
    assert box_dist((5, 5), (0, 0, 10, 10)) == 0 and box_dist((13, 14), (0, 0, 10, 10)) == 5
    assert ref_of("nibs_clinic") == 0 and ref_of("bme_f") == 1
    for b, t in T.items():
        assert t["ref"] == ref_of(b), b  # the reference floor rule never needs truth
    print("pres self-check ok", len(DEV), len(HELD))
