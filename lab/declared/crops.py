"""One crop image per flag, for reading by eye. Usage: crops.py <flags json> [flag id substring]. Writes data/lab/declared/crops/<id>.png."""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/declared")
import cv2  # noqa: E402
import fitz  # noqa: E402
import numpy as np  # noqa: E402
import read_text as T  # noqa: E402

OUT = T.ROOT / "data/lab/declared/crops"


def crop(sid, p, box, pad=36, long_px=1300, min_zoom=1.5, max_zoom=4):
    pg = T.page(sid, p)
    W, H = pg.rect.width, pg.rect.height
    # short strips such as an index row get more context around them
    c = fitz.Rect(max(0, box[0] - pad), max(0, box[1] - pad), min(W, box[2] + pad), min(H, box[3] + pad))
    z = min(max_zoom, max(min_zoom, long_px / max(c.width, c.height)))
    pm = pg.get_pixmap(matrix=fitz.Matrix(z, z), clip=c, alpha=False)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3].copy()
    a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    cv2.rectangle(a, (int((box[0] - c.x0) * z), int((box[1] - c.y0) * z)), (int((box[2] - c.x0) * z), int((box[3] - c.y0) * z)), (0, 0, 230), 2)
    return a


def compose(f):
    parts = []
    head = np.full((46, 1300, 3), 255, np.uint8)
    cv2.putText(head, f["id"][:70], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(head, f["text"][:120], (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 0, 0), 1, cv2.LINE_AA)
    parts.append(head)
    for p, box in f["boxes"][:5]:
        parts.append(crop(f["set"], p, box))
    w = max(x.shape[1] for x in parts)
    parts = [cv2.copyMakeBorder(x, 0, 8, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in parts]
    return np.vstack(parts)


if __name__ == "__main__":
    data = json.loads(open(sys.argv[1]).read())
    OUT.mkdir(parents=True, exist_ok=True)
    for f in data["flags"]:
        if len(sys.argv) > 2 and sys.argv[2] not in f["id"]:
            continue
        if f["boxes"]:
            cv2.imwrite(str(OUT / f"{f['id']}.png"), compose(f))
    print(len(data["flags"]))
