"""Sheet registry and rendering. Core is the 6 dense sheets, holdout is the wider title-block set."""
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
import fitz  # noqa: E402

ROOT = Path(".")
P = ROOT / "data/plans"
OUT = ROOT / "data/lab/bakeoff"
N = "nibs_clinic/2012-03-27-ClinicRedactedDrawings/"
CAP, USC, ADU = "mml_capitol/THE-CAPITOL.pdf", "usc_leconte/USC_LeConte_Bid_Drawings.pdf", "carpinteria_adu1/plan1.pdf"
ABT, BAX, AF = "mml_abbot/THE-ABBOT.pdf", "mml_baxter/THE-BAXTER.pdf", "af_ces_admin/CES_SD_drawings_admin_bldg.pdf"

CORE = {
    "n161": (N + "Clinic_161_Life Safety Composite First Floor Plan.pdf", 0),
    "n79": (N + "Clinic_079_Door Schedule.pdf", 0),
    "u1": (USC, 1),
    "c3": (CAP, 3),
    "hc1": ("habs_stelizabeths_center/02.jpg", None),
    "bf0": ("bme_f/bme_0.jpg", None),
}
EXTRA = {
    "n1": (N + "Clinic_001_Cover Sheet.pdf", 0),
    "n3": (N + "Clinic_003_Index Of Drawings.pdf", 0),
    "n7": (N + "Clinic_007_Site Layout Plan.pdf", 0),
    "n52": (N + "Clinic_052_Composite First Floor Plan.pdf", 0),
    "n53": (N + "Clinic_053_First Floor Plan Block 1A.pdf", 0),
    "n59": (N + "Clinic_059_Composite Second Floor Plan.pdf", 0),
    "n76": (N + "Clinic_076_Finish Schedule.pdf", 0),
    "n91": (N + "Clinic_091_Exterior Building Elevations.pdf", 0),
    "n93": (N + "Clinic_093_Building Sections.pdf", 0),
    "n122": (N + "Clinic_122_Enlarged Stair #1 Floor Plans & Section.pdf", 0),
    "n162": (N + "Clinic_162_Life Safety Composite Second Floor Plan.pdf", 0),
    "n166": (N + "Clinic_166_Foundation And First Floor Plan.pdf", 0),
    "n171": (N + "Clinic_171_Column Schedule.pdf", 0),
    "u0": (USC, 0), "u2": (USC, 2), "u3": (USC, 3), "u12": (USC, 12), "u15": (USC, 15),
    "c0": (CAP, 0), "c2": (CAP, 2), "c4": (CAP, 4), "c5": (CAP, 5), "c7": (CAP, 7),
    "a0": (ADU, 0), "a5": (ADU, 5), "a7": (ADU, 7), "a11": (ADU, 11), "a20": (ADU, 20),
    "ab5": (ABT, 5), "bx4": (BAX, 4), "af0": (AF, 0), "af1": (AF, 1), "af2": (AF, 2),
    "br0": ("brandeis_hall/level-1.pdf", 0),
    "hc0": ("habs_stelizabeths_center/00.jpg", None),
    "ha1": ("habs_stelizabeths_atkins/01.jpg", None),
    "hv1": ("habs_va_bldg1/00.jpg", None),
    "bk1": ("bme_k/00.jpg", None),
    "sc1": ("schenley/sch_first.png", None),
}
ALL = {**CORE, **EXTRA}


def full(key, long_px):
    """Sheet as a BGR array with its long edge at long_px. Raster sheets are never upscaled."""
    f, page = ALL[key]
    if page is None:
        im = cv2.imread(str(P / f))
        s = long_px / max(im.shape[:2])
        return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im
    pg = fitz.open(P / f)[page]
    z = long_px / max(pg.rect.width, pg.rect.height)
    pm = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR).copy()


def text_layer(key):
    f, page = ALL[key]
    return "" if page is None else fitz.open(P / f)[page].get_text()


def title_crops(key, long_px=6000):
    """Fixed crops holding a title block on nearly every set: the right 22% and the bottom 22%."""
    im = full(key, long_px)
    h, w = im.shape[:2]
    return [im[:, int(w * .78):], im[int(h * .78):, :]]


def limit(im, long_px):
    s = long_px / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im


def tiles(im, cols, rows, overlap=.08):
    """Grid crops with overlap. Returns (crop, x0, y0) in pixels of im."""
    h, w = im.shape[:2]
    tw, th = int(w / cols * (1 + overlap)), int(h / rows * (1 + overlap))
    return [(im[y:y + th, x:x + tw], x, y) for x in np.linspace(0, w - tw, cols, dtype=int) for y in np.linspace(0, h - th, rows, dtype=int)]
