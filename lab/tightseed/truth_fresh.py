"""Truth for a stack no model has seen, written before the post-hoc recipe is run on it. East block of Royce Phase 1. No API."""
import hashlib, json, shutil, time
from pathlib import Path
R = Path("/Users/tsharma/Github/Meta/data/lab/tightseed")
def rep(b, fl): return {str(f): b for f in fl}
all4 = range(4)
T = {"_note": "Fresh confirmation stack. Same labeller, by eye from grid views, written before any model call on this window. Same field rules as truth.json.",
     "royce_east": dict(n=4, crop=[1480, 1020, 1960, 1860], split="fresh", source="queens_royce1 frames, window never sent to a model", stairs={
        "E3": dict(cls="main", boxes=rep([1514, 1043, 1552, 1075], all4)),
        "E4": dict(cls="main", boxes=rep([1580, 1225, 1614, 1254], all4)),
        "E5": dict(cls="main", boxes=rep([1580, 1395, 1614, 1422], (0, 1)), note="ends above f1; f2 and f3 say ROOF"),
        "E6": dict(cls="main", boxes=rep([1738, 1491, 1776, 1524], all4)),
        "E7": dict(cls="main", boxes=rep([1580, 1742, 1614, 1771], all4)),
        "F3": dict(cls="short", boxes=rep([1582, 1079, 1620, 1122], (0, 1)), note="corridor flight, ends above f1; roof on f2 and f3"),
        "F4": dict(cls="amb", boxes=rep([1576, 1330, 1595, 1375], all4), note="short flight on f0 and f1, stripes on f2 and f3"),
        "F5": dict(cls="amb", boxes=rep([1576, 1630, 1595, 1672], all4), note="short flight on f0 and f1, stripes on f2 and f3"),
        "S3": dict(cls="amb", boxes=rep([1554, 1036, 1572, 1078], (1, 2, 3))),
        "S4": dict(cls="amb", boxes=rep([1576, 1190, 1612, 1208], (1, 2, 3))),
     })}
p = R / "truth_fresh.json"
if p.exists():
    raise SystemExit("exists")
s = json.dumps(T, indent=1); p.write_text(s)
d = R / "stacks/royce_east"; d.mkdir(exist_ok=True)
for i in range(4):
    shutil.copy(R / f"stacks/queens_royce1/f{i}.png", d / f"f{i}.png")
h = hashlib.sha256(s.encode()).hexdigest(); t = time.strftime("%Y-%m-%dT%H:%M:%S%z")
(R / "truth_fresh.sha256").write_text(f"{h}  truth_fresh.json  written {t}\n")
print(h, t)
