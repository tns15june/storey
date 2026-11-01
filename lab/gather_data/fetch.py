"""Download every file in manifest.json into data/plans/<id>/ and unzip archives there.

Run: .venv/bin/python -I lab/gather_data/fetch.py
Safe to rerun: existing files are skipped. CubiCasa and Swiss Dwellings are resumable curl jobs, see report.md.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLANS = HERE.parent.parent / "data/plans"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"

ZIP_LIMIT = 8 * 1024 ** 3


def safe_extract(zip_path, out):
    """Pull only document files, and reject paths that escape the folder (downloads are untrusted)."""
    with zipfile.ZipFile(zip_path) as z:
        # Declared sizes are checked first and the bytes written are counted, so a zip bomb cannot fill the disk.
        if sum(i.file_size for i in z.infolist()) > ZIP_LIMIT:
            raise ValueError(f"{zip_path} unpacks to more than {ZIP_LIMIT} bytes")
        written = 0
        for info in z.infolist():
            root = out.resolve()
            target = (root / info.filename).resolve()
            if info.is_dir() or not target.is_relative_to(root):
                continue
            if target.suffix.lower() not in {".pdf", ".ifc", ".txt", ".png", ".jpg"} or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if written > ZIP_LIMIT:
                        raise ValueError(f"{zip_path} wrote more than {ZIP_LIMIT} bytes")
                    dst.write(chunk)


def main():
    fails = []
    for item in json.loads((HERE / "manifest.json").read_text()):
        folder = PLANS / item["id"]
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / item["file"]
        if not dest.exists() or dest.stat().st_size == 0:
            r = subprocess.run(["curl", "-sS", "-L", "-f", "-A", UA, "-m", "280", "-o", str(dest), item["url"]])
            if r.returncode:
                fails.append((item["id"], item["file"], r.returncode))
                dest.unlink(missing_ok=True)
                continue
        if dest.suffix == ".zip":
            safe_extract(dest, folder)
    print("failed:", fails)
    sys.exit(0)


main()
