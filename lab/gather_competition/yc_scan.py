"""Filter the public yc-oss dump of YC companies for permit and plan-review names.
Run: /Users/tsharma/Github/Meta/.venv/bin/python -I /Users/tsharma/Github/Meta/lab/gather_competition/yc_scan.py
"""
import json, re, subprocess
from pathlib import Path
OUT = Path("/Users/tsharma/Github/Meta/data/lab/gather_competition")
raw = OUT / "raw/yc_all.json"
if not raw.exists():
    subprocess.run(["curl", "-s", "-m", "60", "-o", str(raw), "https://yc-oss.github.io/api/companies/all.json"], check=True)
rows = json.loads(raw.read_text())
pat = re.compile(r"permit|plan review|plan check|building code|zoning|code compliance|city hall|municipal|construction drawing|blueprint|floor ?plan|architectur(e|al) (firm|drawing)|building department", re.I)
hits = []
for c in rows:
    text = " ".join(str(c.get(k) or "") for k in ("name", "one_liner", "long_description"))
    if pat.search(text):
        hits.append({k: c.get(k) for k in ("name", "batch", "status", "website", "one_liner", "launched_at")})
hits.sort(key=lambda c: c["batch"] or "")
(OUT / "yc_hits.json").write_text(json.dumps(hits, indent=1))
print(len(rows), "companies scanned;", len(hits), "hits")
for c in hits:
    print(f"{c['batch']:<14}{c['status']:<10}{c['name']:<24}{(c['one_liner'] or '')[:95]}  {c['website']}")
