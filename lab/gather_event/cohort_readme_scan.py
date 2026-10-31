#!/usr/bin/env python3
"""Scan README text of in-window code-search hits, because many repos have blank descriptions.

Run after cohort_code_search.py:
/Users/tsharma/Github/Meta/.venv/bin/python /Users/tsharma/Github/Meta/lab/gather_event/cohort_readme_scan.py
Writes: data/lab/gather_event/cohort_readme_scan.json. README text is counted as data, never run.
"""
import base64
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("/Users/tsharma/Github/Meta/data/lab/gather_event")
DRAWING = ["plan review", "plan check", "building code", "building permit", "floor plan", "floorplan", "egress",
           "stairwell", "staircase", "life safety", "ibc", "nfpa", "zoning", "construction drawing",
           "architectural drawing", "engineering drawing"]
META = ["muse-spark", "muse spark", "muse-wonder", "muse wonder", "sam 3", "sam-3", "glimmer", "meta model api",
        "meta global ai"]


def scan(h):
    p = subprocess.run(["gh", "api", "-X", "GET", f"repos/{h['full_name']}/readme"], capture_output=True, text=True)
    if p.returncode != 0:
        return dict(h, readme_chars=0, drawing={}, meta=[])
    t = base64.b64decode(json.loads(p.stdout)["content"]).decode("utf8", "replace").lower()
    return dict(h, readme_chars=len(t), drawing={k: t.count(k) for k in DRAWING if k in t},
                meta=[k for k in META if k in t], head=t[:240])


def main():
    hits = [h for h in json.loads((OUT / "cohort_code_hits.json").read_text()) if h.get("created_at", "") >= "2026-10-20"]
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(scan, hits))
    rows.sort(key=lambda r: -sum(r["drawing"].values()))
    (OUT / "cohort_readme_scan.json").write_text(json.dumps(rows, indent=1))
    print("scanned", len(rows), "with README", sum(1 for r in rows if r["readme_chars"]))
    for r in rows:
        if r["meta"] and sum(r["drawing"].values()) >= 2:
            print(r["created_at"][:10], r["full_name"], sum(r["drawing"].values()), r["drawing"])


if __name__ == "__main__":
    main()
