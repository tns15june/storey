"""Read-only PDF spot checks of two 'real' flags. No API. Run with -I. PDFs are treated as data."""
import fitz, re
D = "data/plans/"
d = fitz.open(D + "mml_abbot/THE-ABBOT.pdf")
print("abbot pages", len(d))
for i, p in enumerate(d):
    t = p.get_text()
    head = " ".join(t.split())[:70]
    rows = [m for m in re.finditer(r"(?m)^\s*20\s*$", t)]
    has = "OVERHEAD" in t.upper() or "GARAGE DOOR" in t.upper()
    if has or len(rows) > 0:
        print(i, "standalone 20 count", len(rows), "garage/overhead words", has, "|", head)
for i, p in enumerate(d):
    t = p.get_text()
    for m in re.finditer(r"(?i)[^\n]*(overhead|garage door|roll.?up)[^\n]*", t):
        print("  p", i, m.group(0)[:110])
for name in ("carpinteria_adu2/plan2.pdf", "carpinteria_adu1/plan1.pdf"):
    d = fitz.open(D + name)
    print(name, "pages", len(d))
    hits = {}
    for i, p in enumerate(d):
        t = p.get_text()
        for k in ("AD-901", "AD-902", "AD-903", "AD-904"):
            n = t.count(k)
            if n:
                hits.setdefault(k, []).append((i, n))
    print(hits)
    labels = []
    for i, p in enumerate(d):
        m = re.findall(r"\b(AD-\d{3})\b", p.get_text())
        if m:
            labels.append((i, sorted(set(m))))
    print(labels[:12])
