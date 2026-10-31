"""Turn an uploaded drawing set into page images plus whatever text and vector content the file holds."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import pymupdf

MAX_EDGE = 6000       # sheets past this add file size, not legibility, at Spark's and SAM's working sizes
RENDER_DPI = 200
VECTOR_MIN_ITEMS = 200   # shortcut: a fixed count separates CAD pages from scans, replace if the lab finds a better signal


@dataclass
class Sheet:
    index: int
    image: str
    width: int
    height: int
    kind: str                # "vector" or "scan"
    text: str = ""
    words: list = field(default_factory=list)   # [x0, y0, x1, y1, word] in image pixels


def _render(page, out):
    scale = min(RENDER_DPI / 72, MAX_EDGE / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    pix.save(out)
    return pix.width, pix.height, scale


def _from_pdf(path, out_dir):
    sheets = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            name = f"sheet_{i:03d}.png"
            w, h, scale = _render(page, out_dir / name)
            kind = "vector" if sum(len(d["items"]) for d in page.get_drawings()) >= VECTOR_MIN_ITEMS else "scan"
            words = [[round(a * scale), round(b * scale), round(c * scale), round(d * scale), t] for a, b, c, d, t, *_ in page.get_text("words")]
            sheets.append(Sheet(i, name, w, h, kind, page.get_text(), words))
    return sheets


def _from_images(paths, out_dir):
    sheets = []
    for i, p in enumerate(sorted(paths)):
        im = cv2.imread(str(p))
        if im is None:
            raise ValueError(f"Could not read {Path(p).name} as an image")
        name = f"sheet_{i:03d}.png"
        cv2.imwrite(str(out_dir / name), im)
        sheets.append(Sheet(i, name, im.shape[1], im.shape[0], "scan"))
    return sheets


def ingest(paths, out_dir):
    """Accepts one PDF or several page images. Writes sheets.json beside the page images and returns the sheets."""
    paths = [Path(p) for p in paths]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = [p for p in paths if p.suffix.lower() == ".pdf"]
    if pdfs and len(paths) > 1:
        raise ValueError("Upload one PDF, or a set of page images, not both")
    sheets = _from_pdf(pdfs[0], out_dir) if pdfs else _from_images(paths, out_dir)
    if not sheets:
        raise ValueError("The upload holds no pages")
    (out_dir / "sheets.json").write_text(json.dumps([asdict(s) for s in sheets]))
    return sheets
