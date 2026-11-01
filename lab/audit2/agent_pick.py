"""Print the case ids for overlays I want to read, and what the D04 row data said. No API."""
import json
from pathlib import Path
D = Path("data/lab/agent")
C = json.load(open(D / "truth.json"))["cases"]
cases = {c["id"]: c for c in json.load(open(D / "cases.json"))}
want = ("door_good_6", "door_good_4", "route_good_1_8", "door_bad_1", "value_agree_0")
for i, c in C.items():
    if c["key"] in want:
        r = json.load(open(D / "runs/spark_low" / f"{i}.json"))
        print(i, c["key"], c["label"], c["cat"], "|", cases[i]["image"], "|", r["out"]["verdict"], r["out"]["reason"][:210])
for n in ("spark_medium_2", "spark_medium_3", "wonder_medium_0"):
    s = json.load(open(D / f"e2e/{n}.score.json"))
    print(n, {k: (v if not isinstance(v, (list, dict)) else str(v)[:300]) for k, v in s.items()})
