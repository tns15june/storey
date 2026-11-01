import warnings; warnings.filterwarnings("ignore")
import fitz, cv2, numpy as np
from pathlib import Path
ROOT = Path("."); OUT = ROOT / "data/lab/tightseed/raw"
d = fitz.open(ROOT / "data/plans/mml_abbot/THE-ABBOT.pdf")
for i, p in enumerate(d):
    t = [x for x in p.get_text().split("\n") if x.strip()]
    title = [x for x in t if any(k in x for k in ("Level", "PLANS", "Foundation", "Module"))][:6]
    print(i, p.rect, title)
px = d[6].get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
cv2.imwrite(str(OUT / "mml_abbot_p6.png"), cv2.cvtColor(np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, 3), cv2.COLOR_RGB2BGR))
