"""Download paper PDFs as data and print lines near benchmark tables.
Run: .venv/bin/python -I lab/gather_competition/papers.py
"""
import re, subprocess
from pathlib import Path
import fitz
D = Path("data/lab/gather_competition/raw/papers")
P = {"aecv":"2601.04819","cubicasa":"1904.01920","floorplancad":"2105.07147","textaided":"2607.12678","textenh":"2510.11091","line":"2505.23395","mitunet":"2512.02413","aecbench":"2603.29199"}
for k,a in P.items():
    f=D/f"{k}.pdf"
    if not f.exists():
        subprocess.run(["curl","-s","-L","-m","60","-o",str(f),f"https://arxiv.org/pdf/{a}"])
    try:
        d=fitz.open(f); t="\n".join(p.get_text() for p in d)
    except Exception as e:
        print(k,"FAILED",e); continue
    (D/f"{k}.txt").write_text(t)
    print(k,a,d.page_count,"pages:",t[:90].replace("\n"," "))
