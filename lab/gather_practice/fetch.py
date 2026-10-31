"""Download the public plan-review documents and extract their text.

Run: /Users/tsharma/Github/Meta/.venv/bin/python -I /Users/tsharma/Github/Meta/lab/gather_practice/fetch.py
Reads urls.tsv (name, url) next to this script. Files already on disk are not fetched again.
Downloads are untrusted: they are only saved and parsed by PyMuPDF, never executed.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "data/reference/practice"
TXT = ROOT / "data/lab/gather_practice/text"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def fetch(name, url):
    path = OUT / name
    if path.exists() and path.stat().st_size > 0:
        return path, "cached"
    r = subprocess.run(["curl", "-sSL", "--compressed", "--http1.1", "--max-time", "25", "-A", UA, "-H", "Accept: text/html,application/pdf,*/*;q=0.8", "-H", "Accept-Language: en-US,en;q=0.9", "-o", str(path), "-w", "%{http_code}", url], capture_output=True, text=True)
    code = r.stdout.strip()
    if code != "200" or not path.exists():
        path.unlink(missing_ok=True)
        return None, f"http {code} {r.stderr.strip()[:80]}"
    return path, "ok"


def to_text(path):
    head = path.read_bytes()[:5]
    if head != b"%PDF-":
        return None, 0
    with fitz.open(path) as doc:
        return "\n".join(f"\n=== page {i + 1} ===\n{p.get_text()}" for i, p in enumerate(doc)), doc.page_count


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TXT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for line in (HERE / "urls.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, url = line.split("\t")
        path, status = fetch(name, url)
        row = {"name": name, "url": url, "status": status}
        if path:
            text, pages = to_text(path)
            row |= {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pages": pages, "is_pdf": text is not None}
            if text is None:
                # shortcut: non-PDF is kept raw for the WebFetch reader, upgrade only if a page matters
                row["chars"] = 0
            else:
                (TXT / f"{name}.txt").write_text(text)
                row["chars"] = len(text)
        manifest.append(row)
        print(name, status, row.get("pages"), row.get("chars"))
    (OUT.parent.parent / "lab/gather_practice/manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
