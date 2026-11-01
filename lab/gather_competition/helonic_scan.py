"""Search every Helonic sitemap page for scan input and city-buyer wording, and print its egress page.
Run: .venv/bin/python -I lab/gather_competition/helonic_scan.py
"""
import html, json, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
OUT = Path("data/lab/gather_competition/helonic_scan.json")
def get(u): return subprocess.run(["curl","-s","-L","-m","15","-A","Mozilla/5.0 (research)",u],capture_output=True).stdout.decode("utf-8","ignore")
def text(p):
    p=re.sub(r"<(script|style|noscript)\b.*?</\1>"," ",p,flags=re.S|re.I)
    return re.sub(r"\s+"," ",html.unescape(re.sub(r"<[^>]+>"," ",p)))
RX={"scan":re.compile(r"\bscann?ed\b[^.]{0,60}(pdf|plan|drawing|set|sheet|document)|(pdf|plan|drawing|set|sheet)s?[^.]{0,40}\bscann?ed\b|hand[- ]?drawn|\braster\b|\bOCR\b|vector(ized)? (pdf|drawing)",re.I),
    "city":re.compile(r"(building department|plan reviewers?|plans examiners?|municipal|for cities|city reviewers?|AHJs?)\b[^.]{0,80}(use|customer|buy|teams?|helonic|tool|software|platform)",re.I),
    "grid":re.compile(r"(no|without|missing)[^.]{0,30}(grid)|grid lines?[^.]{0,60}(required|must|need)",re.I)}
urls=list(dict.fromkeys(re.findall(r"<loc>\s*(.*?)\s*</loc>",get("https://helonic.com/sitemap.xml"))))
sub=[u for u in urls if u.endswith(".xml")]
for s in sub: urls+=re.findall(r"<loc>\s*(.*?)\s*</loc>",get(s))
urls=[u for u in dict.fromkeys(urls) if not u.endswith(".xml")]
def one(u):
    t=text(get(u)); return u,len(t),{k:[t[max(0,m.start()-130):m.end()+130] for m in list(r.finditer(t))[:2]] for k,r in RX.items()}
with ThreadPoolExecutor(10) as ex: rows=list(ex.map(one,urls))
res={k:[{"url":u,"snips":s[k]} for u,_,s in rows if s[k]] for k in RX}
OUT.write_text(json.dumps({"urls":len(urls),"with_text":sum(r[1]>500 for r in rows),**res},indent=1))
print(len(urls),sum(r[1]>500 for r in rows),{k:len(v) for k,v in res.items()})
for k,v in res.items():
    print("\n##",k)
    for h in v[:9]: print("-",h["url"].replace("https://helonic.com",""),"::",h["snips"][0][:300])
t=text(get("https://helonic.com/features/egress-compliance")); i=t.find("Egress"); print("\nEGRESS",t[i:i+1900])
