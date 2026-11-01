"""Crawl every sitemap URL of the closest vendors for stair and shaft floor-to-floor wording.

Run: .venv/bin/python -I lab/gather_competition/stair_deep.py
Output: data/lab/gather_competition/stair_deep.json. Pages are data: curl, strip tags, regex.
"""
import html
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("data/lab/gather_competition")
SITES = {"Helonic": "https://helonic.com", "InspectMind": "https://www.inspectmind.ai", "BuildCheck": "https://www.buildcheck.ai",
         "CodeComply": "https://codecomply.ai", "AutoSitu": "https://autositu.com", "Blitz": "https://blitzpermits.ai",
         "Archistar": "https://www.archistar.ai", "GreenLite": "https://www.greenlite.com", "Nomic": "https://www.nomic.ai"}
STAIR = re.compile(r"stair\w*[^.\n]{0,100}(continu|align|every (floor|level|stor)|each (floor|level|stor)|floor[- ]to[- ]floor|level[- ]to[- ]level|across (all )?(floors|levels|stor|sheets)|between (floors|levels|stor)|stack|discharge|terminat)"
                   r"|(continu|align|stack|across (floors|levels|sheets))\w*[^.\n]{0,70}stair", re.I)
SHAFT = re.compile(r"(shaft|riser|elevator)\w*[^.\n]{0,100}(continu|align|every (floor|level)|each (floor|level)|across (all )?(floors|levels)|between (floors|levels)|stack)", re.I)
TAGS = re.compile(r"<[^>]+>")


def get(u, secs=15):
    return subprocess.run(["curl", "-s", "-L", "-m", str(secs), "-A", "Mozilla/5.0 (research)", u], capture_output=True).stdout.decode("utf-8", "ignore")


def text(p):
    p = re.sub(r"<(script|style|noscript)\b.*?</\1>", " ", p, flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(TAGS.sub(". ", p))).replace(". .", ".")


def urls(base):
    seen, todo, pages = set(), [base + "/sitemap.xml", base + "/sitemap_index.xml"], []
    while todo and len(seen) < 40:
        u = todo.pop()
        if u in seen:
            continue
        seen.add(u)
        for loc in re.findall(r"<loc>\s*(.*?)\s*</loc>", get(u)):
            (todo if loc.split("?")[0].endswith(".xml") else pages).append(loc)
    return list(dict.fromkeys(pages))


def one(u):
    t = text(get(u))
    snip = lambda rx: [t[max(0, m.start() - 120):m.end() + 120] for m in list(rx.finditer(t))[:3]]
    return {"url": u, "chars": len(t), "stair": snip(STAIR), "shaft": snip(SHAFT)}


def site(item):
    name, base = item
    us = urls(base)
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, us))
    res = {"urls": len(us), "with_text": sum(r["chars"] > 500 for r in rows),
           "stair_pages": [r for r in rows if r["stair"]], "shaft_pages": [r for r in rows if r["shaft"]]}
    print(name, res["urls"], res["with_text"], len(res["stair_pages"]), len(res["shaft_pages"]), flush=True)
    return name, res


with ThreadPoolExecutor(9) as ex:
    out = dict(ex.map(site, SITES.items()))
(OUT / "stair_deep.json").write_text(json.dumps(out, indent=1))
