"""Render the never-modelled sets. Plan images are embedded in Queen's PDFs, so also dump the embedded image info. No API."""
import warnings; warnings.filterwarnings("ignore")
import json, sys
import fitz, cv2, numpy as np
from pathlib import Path
ROOT = Path("/Users/tsharma/Github/Meta"); OUT = ROOT / "data/lab/tightseed/raw"
def pix(pg, z):
    px = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
    return cv2.cvtColor(np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, 3), cv2.COLOR_RGB2BGR)
for bid, fn, pages in (("queens_leonard", "leonard_hall.pdf", range(1, 6)), ("queens_royce", "jean_royce_hall.pdf", range(1, 9))):
    d = fitz.open(ROOT / "data/plans" / bid / fn)
    for i in pages:
        pg = d[i]
        print(bid, i, [(x[0], x[2], x[3], x[5]) for x in pg.get_images(full=True)], [tuple(round(v) for v in r) for x in pg.get_images(full=True) for r in pg.get_image_rects(x[0])], pg.rotation)
        cv2.imwrite(str(OUT / f"{bid}_p{i}.png"), pix(pg, 3.0))
d = fitz.open(ROOT / "data/plans/mml_abbot/THE-ABBOT.pdf")
cv2.imwrite(str(OUT / "mml_abbot_p1.png"), pix(d[1], 1.2))
