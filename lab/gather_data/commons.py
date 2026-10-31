"""Fetch whole HABS and BME sets from Wikimedia Commons into data/plans/<id>/ and save exact license metadata.

Run: /Users/tsharma/Github/Meta/.venv/bin/python -I lab/gather_data/commons.py
HABS originals are large TIFFs, so this takes a 5000 px wide server-rendered JPEG. Uses curl because the venv Python has no CA bundle.
"""
import json
import subprocess
import time
import urllib.parse
from pathlib import Path

PLANS = Path(__file__).resolve().parent.parent.parent / "data/plans"
UA = "MetaHackathonResearch/0.1 (plan review research)"
SETS = {
    "habs_stelizabeths_center": 'intitle:"Floor Plan" intitle:"St. Elizabeths Hospital, Center Building"',
    "habs_stelizabeths_atkins": 'intitle:"Plan" intitle:"St. Elizabeths Hospital, Atkins Hall"',
    "habs_va_bldg1": 'intitle:"floor plan as built" intitle:"Veterans Administration Medical Center, Building No. 1"',
    "bme_k": 'intitle:"BME Building K"',
    "schenley": 'intitle:"Schenley High School, 1916" floor plan',
    "bme_f": 'intitle:"BME Building F"',
}


def api(**p):
    time.sleep(2)  # Commons rate-limits bursts
    p.update(action="query", format="json")
    u = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(p)
    return json.loads(subprocess.run(["curl", "-s", "-m", "40", "-A", UA, u], capture_output=True, text=True).stdout)


for bid, query in SETS.items():
    hits = [x["title"] for x in api(list="search", srnamespace=6, srlimit=40, srsearch=query)["query"]["search"]]
    if bid == "habs_stelizabeths_atkins":
        hits = [t for t in hits if "Floor Plan" in t or "Attic Plan" in t]
    tifs = {t[:-4] for t in hits if t.endswith(".tif")}
    hits = [t for t in hits if not (t.endswith(".png") and t[:-4] in tifs)]  # Commons keeps a PNG copy of some TIFFs
    out = PLANS / bid
    out.mkdir(exist_ok=True)
    if (out / "commons_meta.json").exists():
        continue
    meta = []
    for k in range(0, len(hits), 20):
        d = api(titles="|".join(hits[k:k + 20]), prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth=5000,
                iiextmetadatafilter="LicenseShortName|Artist|UsageTerms|DateTimeOriginal|Credit")
        for p in d["query"]["pages"].values():
            i = p["imageinfo"][0]
            e = i.get("extmetadata", {})
            meta.append({"title": p["title"], "page_url": i["descriptionurl"], "license": e.get("LicenseShortName", {}).get("value"),
                         "usage": e.get("UsageTerms", {}).get("value"), "artist": e.get("Artist", {}).get("value", "")[:200],
                         "width": i["width"], "height": i["height"], "mime": i["mime"], "url": i["url"], "thumb": i.get("thumburl")})
    (out / "commons_meta.json").write_text(json.dumps(meta, indent=1))
    if bid in ("schenley", "bme_f"):  # already on disk, only the license record is new
        continue
    for n, m in enumerate(sorted(meta, key=lambda x: x["title"])):
        src = m["thumb"] if m["mime"] == "image/tiff" or m["width"] > 12000 else m["url"]
        dest = out / f"{n:02d}.jpg"
        if not dest.exists():
            r = subprocess.run(["curl", "-sS", "-L", "-f", "-m", "120", "-A", UA, "-o", str(dest), src])
            m["file"] = dest.name if r.returncode == 0 else None
        else:
            m["file"] = dest.name
    (out / "commons_meta.json").write_text(json.dumps(meta, indent=1))
    print(bid, len(meta), [m["file"] for m in meta].count(None), "failed")
