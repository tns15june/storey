"""Checks on real sheets, judged by eye: the 'floor plan' concept on whole sheets, 'circle' at sheet scale, and one corridor on the BME scan stack."""
import subprocess
import sys

import cv2
import fitz
import numpy as np

import core as c
from data import OUT, ROOT

SHEETS = {"capitol_A2": (ROOT / "data/plans/mml_capitol/THE-CAPITOL.pdf", 3), "capitol_gnd": (ROOT / "data/plans/mml_capitol/THE-CAPITOL.pdf", 2),
          "clinic_52": (ROOT / "data/plans/nibs_clinic/2012-03-27-ClinicRedactedDrawings/Clinic_052_Composite First Floor Plan.pdf", 0)}


def page(path, n, w=2000):
    p = fitz.open(path)[n]
    pm = p.get_pixmap(matrix=fitz.Matrix(w / p.rect.width, w / p.rect.width))
    return cv2.cvtColor(np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[..., :3], cv2.COLOR_RGB2BGR)


def frame(n):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(ROOT / "data/plans/floors.mp4"), "-vf", f"select=eq(n\\,{n})", "-fps_mode", "passthrough", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


if __name__ == "__main__":
    if sys.argv[1] == "frame":
        f = frame(11)
        print(f.shape)
        cv2.imwrite(str(OUT / "bme_frame11.jpg"), f)
    if sys.argv[1] == "sheets":
        t = []
        for k, (p, n) in SHEETS.items():
            im = page(p, n)
            r = c.sam("floor plan", im)
            v = im.copy()
            for _, b, cf, mk in r:
                v = c.overlay(v, mk)
            print(k, "floor plan masks", len(r), [(b, round(cf or 0, 2)) for _, b, cf, _ in r])
            t.append((v, f"{k} floor plan {len(r)}"))
        c.sheet(t, OUT / "real_sheets.jpg", cols=3, w=640)
    if sys.argv[1] == "bme":
        w, h = (int(v) for v in sys.argv[2].split("x"))
        box = tuple(int(v) for v in sys.argv[3].split(","))
        p = ROOT / "data/plans/floors.mp4"
        res = c.video(c.obj_prompt([dict(box=box)], w, h, frame=8), p, 16)
        t = []
        for i, lv in enumerate(["level 2", "level 1", "level 0 PROMPT", "level -1"]):
            f = frame(i * 4 + 3)
            r = res.get((i * 4 + 3, 1))
            v = c.overlay(f, c.paste(r[0], r[2], f.shape)) if r else f
            print(lv, r[0] if r else None, round(r[1] or 0, 2) if r else None)
            t.append((v, lv))
        c.sheet(t, OUT / "bme_corridor.jpg", cols=4, w=480)
    print("spent", round(c.m.spent("samcraft"), 3))
