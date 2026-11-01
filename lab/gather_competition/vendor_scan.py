"""Crawl vendor sitemaps and search page text for advertised capabilities.

Run: .venv/bin/python -I lab/gather_competition/vendor_scan.py
Output: data/lab/gather_competition/vendor_scan.json and vendor_scan_hits.md.
Pages are treated as data: fetched with curl, tags stripped, regex only. A page that needs JavaScript
shows little text, so `chars` is reported per vendor. Zero hits means not advertised on crawled text, nothing more.
"""
import html
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("data/lab/gather_competition")
CAP = 150
VENDORS = {
    "Archistar": ["https://www.archistar.ai"],
    "CodeComply": ["https://codecomply.ai"],
    "InspectMind": ["https://www.inspectmind.ai"],
    "Blitz": ["https://blitzpermits.ai"],
    "BuildCheck": ["https://www.buildcheck.ai"],
    "Helonic": ["https://helonic.com"],
    "GreenLite": ["https://www.greenlite.com"],
    "AutoSitu": ["https://autositu.com"],
    "Verdant": ["https://www.verdantapp.com"],
    "Nomic": ["https://www.nomic.ai"],
    "CivCheck": ["https://www.civcheck.ai"],
    "Symbium": ["https://www.symbium.com"],
    "UpCodes": ["https://up.codes"],
    "Solibri": ["https://www.solibri.com"],
    "Mason": ["https://www.getmason.ai", "https://www.masonai.com"],
}
PRIORITY = re.compile(r"egress|stair|life-?safety|fire|plan-?review|plan-?check|precheck|check|review|compliance|multifamily|resubmi|revision|correct|product|feature|faq|government|scan|how-it", re.I)
TOPICS = {
    "stair_or_shaft_continuity": r"(stair\w*|shafts?|elevators?|risers?)[^.\n]{0,70}(continu|align|every (floor|level|stor)|each (floor|level|stor)|floor[- ]to[- ]floor|level[- ]to[- ]level|across (all )?(floors|levels|stor)|between (floors|levels|stor)|terminat)|(continu|align)\w*[^.\n]{0,50}(stair|shaft)|vertical (continuity|alignment|circulation)",
    "any_cross_floor": r"(across|between) (the )?(floors|levels|stories|storeys)|floor[- ]to[- ]floor|level[- ]to[- ]level|stack (alignment|conflict)|stacked (plans|floors|levels)",
    "scan_input": r"\bscann?ed\b[^.\n]{0,50}(pdf|plan|drawing|set|sheet|document)|hand[- ]?drawn|raster(ized)? (pdf|plan|drawing|image)|non[- ]?vector|flattened pdf|vector(ized)? pdf|\bocr\b",
    "cycle2_verification": r"(re-?submi\w*|back-?check\w*|re-?check\w*|revis\w+|second (cycle|round|review)|prior (comment|correction)\w*|previous (comment|correction)\w*)[^.\n]{0,90}(resolv|addressed|closed|fixed|verif|confirm|track|cleared|re-?analy)",
    "egress_measure": r"travel distance|common path|dead[- ]end|egress (width|capacity|path)|exit (separation|capacity|access)|occupant load",
    "input_bim_ifc": r"\bIFC\b|\bRevit\b|\bBIM\b",
    "no_approval_disclaimer": r"(not|never|doesn.t) (approve|certif|replace)|final (approval|decision|determination|authority)|humans? in the loop|human[- ]in[- ]the[- ]loop|licensed (reviewer|engineer|architect)",
    "accuracy_number": r"\b\d{2,3}(\.\d)?\s?%[^.\n]{0,30}(accura|precision|recall|detect)|(accura|precision|recall)\w*[^.\n]{0,30}\b\d{2,3}(\.\d)?\s?%",
}
TOPICS = {k: re.compile(v, re.I) for k, v in TOPICS.items()}


def get(url, secs=15):
    r = subprocess.run(["curl", "-s", "-L", "-m", str(secs), "-A", "Mozilla/5.0 (research)", url], capture_output=True)
    return r.stdout.decode("utf-8", "ignore")


def text_of(page):
    page = re.sub(r"<(script|style|noscript)\b.*?</\1>", " ", page, flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", ". ", page))).replace(". .", ".")


def sitemap_urls(base):
    seen, todo, pages = set(), [base + "/sitemap.xml", base + "/sitemap_index.xml", base + "/sitemap-index.xml"], []
    while todo and len(seen) < 40:
        u = todo.pop()
        if u in seen:
            continue
        seen.add(u)
        for loc in re.findall(r"<loc>\s*(.*?)\s*</loc>", get(u)):
            (todo if loc.split("?")[0].endswith(".xml") else pages).append(loc)
    return list(dict.fromkeys(pages))


def scan_vendor(item):
    name, bases = item
    base = bases[0]
    urls = []
    for b in bases:
        urls = sitemap_urls(b)
        if urls:
            base = b
            break
    total = len(urls)
    urls = [base + "/"] + sorted(urls, key=lambda u: (not PRIORITY.search(u), len(u)))[:CAP]

    def one(u):
        t = text_of(get(u))
        return u, len(t), {k: [t[max(0, m.start() - 110):m.end() + 110] for m in list(p.finditer(t))[:3]] for k, p in TOPICS.items()}

    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, urls))
    res = {"sitemap_urls": total, "fetched": len(rows), "chars": sum(r[1] for r in rows), "pages_with_text": sum(r[1] > 500 for r in rows), "topics": {}}
    for k in TOPICS:
        hits = [{"url": u, "snips": s[k]} for u, _, s in rows if s[k]]
        res["topics"][k] = {"pages": len(hits), "examples": hits[:6]}
    print(name, res["sitemap_urls"], res["fetched"], res["pages_with_text"], {k: v["pages"] for k, v in res["topics"].items()}, flush=True)
    return name, res


with ThreadPoolExecutor(5) as ex:
    all_res = dict(ex.map(scan_vendor, VENDORS.items()))
(OUT / "vendor_scan.json").write_text(json.dumps(all_res, indent=1))
lines = []
for name, res in all_res.items():
    lines.append(f"## {name}: {res['fetched']} pages fetched of {res['sitemap_urls']} in sitemap, {res['pages_with_text']} with text")
    for k, v in res["topics"].items():
        lines.append(f"- {k}: {v['pages']} pages")
        for ex_ in v["examples"][:3]:
            lines.append(f"  - {ex_['url']}: {ex_['snips'][0]}")
(OUT / "vendor_scan_hits.md").write_text("\n".join(lines))
