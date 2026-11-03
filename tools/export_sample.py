"""Copies one finished review into a static sample folder: one review per jurisdiction and every file those reviews name.

    PYTHONPATH=. .venv/bin/python tools/export_sample.py <review folder, hex-named> app/web/sample/<id>

The review folder is assembled once per jurisdiction (evaluate and assemble only, no model call) and left assembled for Houston.
sheets.json keeps index, width and height alone, the fields the static screen reads.
"""
import json
import re
import shutil
import sys
from pathlib import Path

from storey.assemble import assemble

src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
assert re.fullmatch(r"[0-9a-f]{32}", src.name), "assemble names files by the folder, and the static screen rewrites only hex ids"
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)
names = set()
for city in ("houston", "philadelphia", "seattle"):
    assemble(src, city)
    text = (src / "review.json").read_text()
    (out / f"review.{city}.json").write_text(text)
    names |= set(re.findall(rf"/api/reviews/{src.name}/files/([^\"?#]+)", text))
assemble(src, "houston")
for name in sorted(names):
    target = (out / name).resolve()
    assert target.is_relative_to(out), name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / name, target)
shutil.copyfile(src / "stage_floors.json", out / "stage_floors.json")
sheets = json.loads((src / "sheets.json").read_text())
(out / "sheets.json").write_text(json.dumps([{k: s[k] for k in ("index", "width", "height")} for s in sheets]))
print(out.name, len(names), "files")
