"""Find what the agent runs saw for row D04. No API."""
import json, re
from pathlib import Path
D = Path("data/lab/agent")
for n in ("wonder_medium_0", "spark_medium_2"):
    s = json.load(open(D / f"e2e/{n}.json"))
    print(n, type(s), list(s)[:10] if isinstance(s, dict) else len(s))
    txt = json.dumps(s)
    for m in list(re.finditer(r"D04", txt))[:6]:
        print("   ", txt[max(0, m.start() - 220):m.end() + 320].replace("\\n", " ")[:540])
