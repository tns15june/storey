"""Letter test: same findings and prompt to each model, plus a letter built in code. Checks that no number, section or sheet appears that is not in the findings.

Run: env LAB_NAME=agent LAB_BUDGET=5 LAB_TOTAL=75 .venv/bin/python lab/agent/letters.py
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/agent")
import metalab as m
import letter_data as L

D = Path("/Users/tsharma/Github/Meta/data/lab/agent/letters"); D.mkdir(exist_ok=True)
SRC = json.dumps(L.FINDINGS) + " " + json.dumps(L.HEADER)
NUM = re.compile(r"\d+(?:\.\d+)*")
SHEET = re.compile(r"\b[A-Z]\d{0,2}[-.]\d+(?:\.\d+)?\b")
SECTION = re.compile(r"\bCBC\s*(?:Section\s*|Table\s*)?([0-9A-Z][0-9A-Z.\-]*)")


def strip_numbering(line):
    return re.sub(r"^\s*(?:\d+|[a-z])[.)]\s+", "", line)


def check(text):
    body = "\n".join(strip_numbering(x) for x in text.splitlines())
    allowed = set(NUM.findall(SRC))
    extra_num = sorted({n for n in NUM.findall(body) if n not in allowed})
    extra_sheet = sorted({s for s in SHEET.findall(body) if s not in SRC})
    extra_sec = sorted({s.rstrip(".") for s in SECTION.findall(body) if s.rstrip(".") not in SRC})
    ban = sorted({w for w in re.findall(r"(?i)\b(approved?|complies|compliant|acceptable|passes|passed)\b", body)})
    leak = [w for w in ("Sea Ranch", "Sonoma", "BLD", "three (3)", "Gallery", "Retail", "Dishwash", "11B") if w in text]
    secs = [f["section"] for f in L.FINDINGS if f["section"]]
    missing_values = []
    for f in L.FINDINGS:
        for k in ("printed", "measured"):
            for n in NUM.findall(f[k]):
                if n not in NUM.findall(body):
                    missing_values.append(f"{f['id']}:{n}")
    return dict(extra_numbers=extra_num, extra_sheets=extra_sheet, extra_sections=extra_sec, banned=ban, leaked=leak, sections_present=sum(s in text for s in set(secs)), n_sections=len(set(secs)),
                sheets_present=sum(f["sheet"] in text for f in L.FINDINGS), missing_values=missing_values, words=len(text.split()))


def templated():
    out = [f"Subject: {L.HEADER['subject']}", f"Project: {L.HEADER['project']}", ""]
    sheets = []
    for f in L.FINDINGS:
        if f["sheet"] not in sheets:
            sheets.append(f["sheet"])
    for i, s in enumerate(sheets, 1):
        out.append(f"{i}. Sheet {s}")
        for j, f in enumerate([x for x in L.FINDINGS if x["sheet"] == s]):
            vals = "; ".join(v for v in (f"printed {f['printed']}" if f["printed"] else "", f"measured {f['measured']}" if f["measured"] else "") if v)
            out.append(f" {'abcdefgh'[j]}. {f['correction']} at {f['location']} ({vals}).{' (' + f['section'] + ')' if f['section'] else ''}")
    return "\n".join(out)


if __name__ == "__main__":
    prompt = L.PROMPT.format(subject=L.HEADER["subject"], project=L.HEADER["project"], example=L.EXAMPLE, findings=json.dumps(L.FINDINGS, indent=1))
    res = {}
    t = templated()
    (D / "template.txt").write_text(t)
    res["template"] = dict(check(t), usd=0.0, seconds=0.0)
    for key, model in (("spark", "muse-spark-1.3"), ("wonder", "muse-wonder-1.0")):
        for run in range(2):
            p = D / f"{key}_{run}.txt"
            q = D / f"{key}_{run}.json"
            if not q.exists():
                text, info = m.spark(prompt, model=model, effort="medium", max_tokens=6000)
                p.write_text(text or "")
                json.dump(info, open(q, "w"))
            info = json.load(open(q))
            res[f"{key}_{run}"] = dict(check(p.read_text()), usd=round(info["usd"], 4), seconds=round(info["seconds"], 1), output=info["output"])
    json.dump(res, open(D / "scores.json", "w"), indent=1)
    for k, v in res.items():
        print(k, json.dumps(v))
    print("agent spent", round(m.spent("agent"), 4))
