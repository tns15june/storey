"""Regenerate declared's frozen flags from frozen code and cached reads, output redirected to audit2. LAB_BUDGET=0 stops any model call. Run with -I."""
import json, runpy, sys
from pathlib import Path
sys.path.insert(0, "lab"); sys.path.insert(0, "lab/declared")
import checks as C
C.OUT = Path("data/lab/audit2/declared_flags")
scope = sys.argv[1]
sys.argv = ["run.py", scope, "_v2"]
runpy.run_path("lab/declared/run.py", run_name="__main__")
a = json.load(open(f"data/lab/declared/flags_{scope}_v2.json"))["flags"]
b = json.load(open(C.OUT / f"flags_{scope}_v2.json"))["flags"]
ia = {f["id"]: f["text"] for f in a}; ib = {f["id"]: f["text"] for f in b}
print(scope, "frozen", len(ia), "rerun", len(ib), "identical", ia == ib, "only frozen", sorted(set(ia) - set(ib))[:5], "only rerun", sorted(set(ib) - set(ia))[:5])
