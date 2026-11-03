"""Turn an uploaded drawing set into page images plus whatever text and vector content the file holds."""
import json
import resource
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import pymupdf
from PIL import Image

MAX_EDGE = 6000       # sheets past this add file size, not legibility, at Spark's and SAM's working sizes
MAX_SHEETS = 300      # a tiny PDF can hold thousands of pages, and each page costs render time and API money
MAX_PIXELS = 150_000_000   # a small image file can decode to gigabytes, so check the header first
MAX_WORDS = 50_000    # per page, for the same reason
MAX_DISK_BYTES = 8 * 1024 ** 3
CPU_SECONDS = 600
MEMORY_BYTES = 8 * 1024 ** 3
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
    sheets, written = [], 0
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("This PDF is password protected")
        if doc.page_count > MAX_SHEETS:
            raise ValueError(f"This PDF has {doc.page_count} pages, and the limit is {MAX_SHEETS}")
        for i, page in enumerate(doc):
            name = f"sheet_{i:03d}.png"
            w, h, scale = _render(page, out_dir / name)
            written += (out_dir / name).stat().st_size
            if written > MAX_DISK_BYTES:
                raise ValueError("This PDF is too large to read")
            kind = "vector" if sum(len(d["items"]) for d in page.get_drawings()) >= VECTOR_MIN_ITEMS else "scan"
            words = [[round(a * scale), round(b * scale), round(c * scale), round(d * scale), t] for a, b, c, d, t, *_ in page.get_text("words")[:MAX_WORDS]]
            sheets.append(Sheet(i, name, w, h, kind, page.get_text(), words))
    return sheets


def _from_images(paths, out_dir):
    if len(paths) > MAX_SHEETS:
        raise ValueError(f"This upload has {len(paths)} images, and the limit is {MAX_SHEETS}")
    sheets, written = [], 0
    for i, p in enumerate(sorted(paths)):
        try:
            with Image.open(p) as probe:     # reads the header only
                pixels = probe.width * probe.height
        except OSError:
            raise ValueError(f"Could not read {Path(p).name} as an image") from None
        if pixels > MAX_PIXELS:
            raise ValueError(f"{Path(p).name} is too large to read safely")
        im = cv2.imread(str(p))
        if im is None:
            raise ValueError(f"Could not read {Path(p).name} as an image")
        if max(im.shape[:2]) > MAX_EDGE:
            k = MAX_EDGE / max(im.shape[:2])
            im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        name = f"sheet_{i:03d}.png"
        cv2.imwrite(str(out_dir / name), im)
        written += (out_dir / name).stat().st_size
        if written > MAX_DISK_BYTES:
            raise ValueError("These page images are too large to read")
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
    try:
        sheets = _from_pdf(pdfs[0], out_dir) if pdfs else _from_images(paths, out_dir)
    except BaseException:
        for sheet in out_dir.glob("sheet_*.png"):   # a refused upload must not keep the disk it filled
            sheet.unlink()
        raise
    if not sheets:
        raise ValueError("The upload holds no pages")
    (out_dir / "sheets.json").write_text(json.dumps([asdict(s) for s in sheets]))
    return sheets


def main(argv):
    """Runs in a child process, so a hostile file can be killed. MuPDF work in a thread cannot be interrupted."""
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    except (ValueError, OSError):
        pass   # shortcut: macOS refuses this limit, the wall-clock timeout in the server still applies
    out_dir, *paths = argv
    result = Path(out_dir) / "ingest_result.json"   # a file, because a hostile PDF can flood stdout with library warnings
    try:
        result.write_text(json.dumps({"sheets": len(ingest(paths, out_dir))}))
    except ValueError as e:
        result.write_text(json.dumps({"error": str(e)}))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
