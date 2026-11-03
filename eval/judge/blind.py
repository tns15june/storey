"""Copies each match sheet without its method, so a judge cannot tell Storey's claims from the baseline's, and writes the key to undo it.

    python blind.py <repo>/data/eval/sheets <out dir>

<out dir>/sheet_NN.json holds set, how, claims and truth. <out dir>/key.json maps NN to the sheet's set, method, path and sha256.
"""
import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
key = {}
for n, path in enumerate(sorted(src.glob("*.json")), 1):
    sheet = json.loads(path.read_text())
    if not sheet["truth"]:
        continue   # nothing to map to: the scorer needs no judge here
    name = f"sheet_{n:02d}"
    (out / f"{name}.json").write_text(json.dumps({k: sheet[k] for k in ("set", "how", "claims", "truth")}, indent=1))
    key[name] = {"set": sheet["set"], "method": sheet["method"], "path": str(path), "sha256": sheet["sha256"], "claims": len(sheet["claims"]), "truth": len(sheet["truth"])}
(out / "key.json").write_text(json.dumps(key, indent=1))
print(json.dumps(key, indent=1))
