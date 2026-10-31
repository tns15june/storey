#!/usr/bin/env python3
"""Read-only GitHub code search for cohort entries that review drawings or permits.

The repository search in cohort_search.py sees only name, description and topics, so it misses
entries whose domain appears only in the README. This script searches file contents.

Run: /Users/tsharma/Github/Meta/.venv/bin/python /Users/tsharma/Github/Meta/lab/gather_event/cohort_code_search.py
Writes: data/lab/gather_event/cohort_code_hits.json and cohort_code_queries.tsv.
"""
import json
import subprocess
import time
import urllib.parse
from pathlib import Path

OUT = Path("/Users/tsharma/Github/Meta/data/lab/gather_event")
DOMAIN = ['"plan review"', '"plan check"', '"plan checker"', '"building code"', '"building permit"', '"permit review"',
          '"permit application"', '"floor plan"', '"floorplan"', '"blueprint"', 'egress', 'zoning', '"city hall"',
          '"fire code"', '"life safety"', '"architectural drawing"', '"construction drawings"', '"code compliance"',
          '"engineering drawing"', '"accessibility compliance"']
ANCHORS = ['"muse-spark"', '"Meta Global AI Developer Hackathon"']


def search(q):
    url = "search/code?per_page=100&q=" + urllib.parse.quote(q)
    for _ in range(3):
        p = subprocess.run(["gh", "api", "-X", "GET", url], capture_output=True, text=True)
        if p.returncode == 0:
            return json.loads(p.stdout), ""
        if "rate limit" in p.stderr.lower() or "403" in p.stderr:
            time.sleep(45)
            continue
        return None, p.stderr.strip()[:160]
    return None, "gave up after rate limit"


def main():
    repos, log = {}, []
    queries = [f"{d} {a}" for a in ANCHORS for d in DOMAIN]
    for i, q in enumerate(queries):
        data, err = search(q)
        n = data["total_count"] if data else -1
        log.append(f"{q}\t{n}\t{err}")
        for item in (data or {}).get("items", []):
            r = item["repository"]
            h = repos.setdefault(r["full_name"], {"full_name": r["full_name"], "description": r["description"],
                                                  "url": r["html_url"], "queries": [], "paths": []})
            h["queries"].append(q)
            if item["path"] not in h["paths"]:
                h["paths"].append(item["path"])
        time.sleep(6.5)  # code search allows 10 requests a minute
        print(i + 1, len(queries), q, n, flush=True)
    (OUT / "cohort_code_queries.tsv").write_text("query\ttotal_count\terror\n" + "\n".join(log) + "\n")

    # Fill in dates for each repo so entries from before the event can be dropped.
    for h in repos.values():
        p = subprocess.run(["gh", "api", "-X", "GET", f"repos/{h['full_name']}"], capture_output=True, text=True)
        if p.returncode == 0:
            d = json.loads(p.stdout)
            h.update(created_at=d["created_at"], pushed_at=d["pushed_at"], stars=d["stargazers_count"])
        time.sleep(0.15)
    rows = sorted(repos.values(), key=lambda h: -len(h["queries"]))
    (OUT / "cohort_code_hits.json").write_text(json.dumps(rows, indent=1))
    print("unique repos", len(rows))


if __name__ == "__main__":
    main()
