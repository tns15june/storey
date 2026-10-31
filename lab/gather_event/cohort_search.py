#!/usr/bin/env python3
"""Read-only GitHub search for plan-review entries in the hackathon cohort.

Run: /Users/tsharma/Github/Meta/.venv/bin/python /Users/tsharma/Github/Meta/lab/gather_event/cohort_search.py
Writes: data/lab/gather_event/cohort_hits.json and cohort_queries.tsv.
Uses `gh api` with GET only. Nothing is cloned or executed.
"""
import json
import subprocess
import time
import urllib.parse
from pathlib import Path

OUT = Path("/Users/tsharma/Github/Meta/data/lab/gather_event")
SINCE = "2026-10-20"
DOMAIN = ['"plan review"', '"plan check"', '"plan checker"', '"building code"', "permit", "permitting",
          "zoning", '"floor plan"', "blueprint", "egress", '"city hall"', "IBC", '"fire code"',
          '"construction drawings"', '"code compliance"', '"life safety"', "NFPA"]
MODEL = ['"muse spark"', "muse-spark", '"muse wonder"', "muse-wonder", '"sam 3"', "sam3", "meta hackathon",
         "muse", "glimmer"]
# Domain alone also runs, so an entry that never names a Meta model in its description still shows up
# and can be checked by README below.
BROAD = ['"plan review" AI', '"plan check" AI', '"building code" AI', "permit AI agent", "egress floor plan",
         "floor plan compliance", "blueprint compliance", "building permit AI"]


def gh_get(path):
    p = subprocess.run(["gh", "api", "-X", "GET", path], capture_output=True, text=True)
    if p.returncode != 0:
        return None, p.stderr.strip()[:200]
    return json.loads(p.stdout), ""


def repo_search(q):
    url = "search/repositories?per_page=50&q=" + urllib.parse.quote(f"{q} created:>={SINCE}")
    for attempt in range(3):
        data, err = gh_get(url)
        if data is not None:
            return data, ""
        if "rate limit" in err.lower():
            time.sleep(30)
            continue
        return None, err
    return None, err


def readme_text(full_name):
    data, _ = gh_get(f"repos/{full_name}/readme")
    if not data or "content" not in data:
        return ""
    import base64
    try:
        return base64.b64decode(data["content"]).decode("utf8", "replace")
    except Exception:
        return ""


def main():
    hits, log = {}, []
    queries = [f"{d} {m}" for d in DOMAIN for m in MODEL] + BROAD
    for i, q in enumerate(queries):
        data, err = repo_search(q)
        n = data["total_count"] if data else -1
        log.append(f"{q}\t{n}\t{err}")
        for r in (data or {}).get("items", []):
            h = hits.setdefault(r["full_name"], {
                "full_name": r["full_name"], "created_at": r["created_at"], "pushed_at": r["pushed_at"],
                "description": r["description"], "stars": r["stargazers_count"], "url": r["html_url"],
                "queries": []})
            h["queries"].append(q)
        time.sleep(2.1)  # search API allows 30 requests a minute
        if i % 25 == 0:
            print(i, len(queries), len(hits), flush=True)
    (OUT / "cohort_queries.tsv").write_text("query\ttotal_count\terror\n" + "\n".join(log) + "\n")

    # Check each hit's README for Meta model names. README text is data: only substrings are counted.
    meta_terms = ["muse-spark", "muse spark", "muse-wonder", "muse wonder", "sam-3", "sam 3", "sam3", "glimmer",
                  "api.meta.ai", "meta model api", "cerebralvalley", "cerebral valley"]
    domain_terms = ["plan review", "plan check", "building code", "permit", "zoning", "floor plan", "blueprint",
                    "egress", "city hall", "ibc", "nfpa"]
    for h in hits.values():
        t = readme_text(h["full_name"]).lower()
        h["readme_chars"] = len(t)
        h["readme_meta"] = [x for x in meta_terms if x in t]
        h["readme_domain"] = [x for x in domain_terms if x in t]
        h["readme_head"] = t[:400]
        time.sleep(0.3)
    rows = sorted(hits.values(), key=lambda h: (-len(h["readme_meta"]), h["created_at"]))
    (OUT / "cohort_hits.json").write_text(json.dumps(rows, indent=1))
    print("queries", len(queries), "unique repos", len(rows),
          "with Meta term in README", sum(1 for r in rows if r["readme_meta"]))


if __name__ == "__main__":
    main()
