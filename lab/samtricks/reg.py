"""Sheet registry for samtricks. Renders use the bakeoff renderer, so pixels match earlier rounds. No API calls here."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, "lab/bakeoff")
import sheets as s  # noqa: E402

ROOT = Path(".")
OUT = ROOT / "data/lab/samtricks"
SHEETS = OUT / "sheets"
TRUTH = OUT / "truth"
N = s.N
BCOE = "../reference/practice/bcoe_center_addendum3.pdf"

s.ALL.update({
    "n122b": (N + "Clinic_122_Enlarged Stair #1 Floor Plans & Section.pdf", 0),
    "n133": (N + "Clinic_133_Enlarged Wet Areas - Toilet Plans.pdf", 0),
    "c11": (s.CAP, 11), "c14": (s.CAP, 14), "c16": (s.CAP, 16),
    "b2": (s.BAX, 2), "b9": (s.BAX, 9), "b14": (s.BAX, 14), "b18": (s.BAX, 18),
    "u7": (s.USC, 7), "u9": (s.USC, 9), "u11": (s.USC, 11),
    "a8": (s.ADU, 8), "a9": (s.ADU, 9),
    "hv2": ("habs_va_bldg1/02.jpg", None),
    "bk2": ("bme_k/01.jpg", None),
    "bcoe1": (BCOE, 1), "bcoe2": (BCOE, 2), "bcoe3": (BCOE, 3),
})

# Item 1. The split is fixed here, before any model call. Dev sets the SAM conf floor, held-out only scores.
DEV = ["n52", "c3", "u3", "a7", "af1", "hv1", "c14", "n91", "n122b", "bk1"]
HELD = ["n161", "n79", "n133", "c11", "c16", "c7", "b2", "b9", "b14", "b18", "u2", "u9", "u11", "u7", "a8", "a9", "a11", "af2", "hv2", "bk2"]
SET_OF = {"n": "nibs_clinic", "c": "mml_capitol", "b": "mml_baxter", "u": "usc_leconte", "a": "carpinteria_adu1", "af": "af_ces_admin", "hv": "habs_va_bldg1", "bk": "bme_k", "bcoe": "bcoe_center", "hc": "habs_stelizabeths_center"}
# Item 2.
WORD_SHEETS = ["bcoe1", "bcoe2", "bcoe3", "af1", "af2", "u2", "u9", "a8", "n161", "n79", "hc1", "c3"]
WORDS = ["title block", "table", "schedule", "north arrow", "scale bar", "stamp", "revision cloud", "legend"]


def sheet(key):
    """The 2000 px sheet every model and every truth box refers to."""
    p = SHEETS / f"{key}.png"
    if not p.exists():
        SHEETS.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), s.full(key, 2000))
    return cv2.imread(str(p))


if __name__ == "__main__":
    assert len(DEV) == 10 and len(HELD) == 20 and not set(DEV) & set(HELD)
    for k in DEV + HELD + WORD_SHEETS:
        print(k, s.ALL[k][0][-40:], s.ALL[k][1], sheet(k).shape)
