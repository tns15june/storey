"""Per-case verdicts for every config, same-subset scores and reasons. No API."""
import json, math, sys
from pathlib import Path
D = Path("data/lab/agent")
T = json.load(open(D / "truth.json"))["cases"]
CF = ["spark_low", "spark_medium", "wonder_low", "wonder_medium", "spark_low_blind"]
R = {c: {p.stem: json.load(open(p)) for p in (D / "runs" / c).glob("case_*.json")} for c in CF}
flag = lambda r: r and r["out"] and r["out"]["verdict"] == "defect"
print("case      cat   lab src  key                    " + " ".join(f"{c[:9]:>9}" for c in CF))
for cid, t in sorted(T.items(), key=lambda kv: (kv[1]["label"], kv[1]["cat"], kv[1]["source"])):
    print(f"{cid} {t['cat']:5s} {t['label']:4s} {t['source'][:4]} {t['key']:22s} " + " ".join(f"{('D' if flag(R[c].get(cid)) else '.') if cid in R[c] else '-':>9}" for c in CF))
first16 = [f"case_{i:02d}" for i in range(16)]
print("\nsame first 16 cases (6 bad, 10 good):")
for c in CF[:4]:
    bad = [i for i in first16 if T[i]["label"] == "bad"]; good = [i for i in first16 if T[i]["label"] == "good"]
    print(c, "caught", sum(bool(flag(R[c][i])) for i in bad), "/", len(bad), "false rej", sum(bool(flag(R[c][i])) for i in good), "/", len(good), "usd", round(sum(R[c][i]["usd"] for i in first16), 3))
print("\nwithout routes (all 32 less 8 route cases):")
for c in CF[:3]:
    bad = [i for i in T if T[i]["label"] == "bad" and T[i]["cat"] != "route"]; good = [i for i in T if T[i]["label"] == "good" and T[i]["cat"] != "route"]
    print(c, "caught", sum(bool(flag(R[c][i])) for i in bad), "/", len(bad), "false rej", sum(bool(flag(R[c][i])) for i in good), "/", len(good))
print("\nand-gate Spark low and Wonder low (flag only if both flag):")
bad = [i for i in T if T[i]["label"] == "bad"]; good = [i for i in T if T[i]["label"] == "good"]
print(" caught", sum(bool(flag(R["spark_low"][i]) and flag(R["wonder_low"][i])) for i in bad), "/16 false rej", sum(bool(flag(R["spark_low"][i]) and flag(R["wonder_low"][i])) for i in good), "/16")
print("\nreasons on good cases flagged, and on missed bad cases:")
for c in CF[:4]:
    for i, r in sorted(R[c].items()):
        t = T[i]
        if (t["label"] == "good" and flag(r)) or (t["label"] == "bad" and not flag(r)):
            print(f"[{c}] {t['key']} {t['label']} ({r['out']['x']},{r['out']['y']}): {r['out']['reason'][:210]}")
print("\nroute catches:")
for c in CF[:4]:
    for i, r in sorted(R[c].items()):
        t = T[i]
        if t["cat"] == "route" and t["label"] == "bad" and flag(r):
            d = min(math.dist((r["out"]["x"], r["out"]["y"]), q[:2]) for q in t["defect"])
            print(f"[{c}] {t['key']} truth {t['defect']} said ({r['out']['x']},{r['out']['y']}) dist {d:.0f}: {r['out']['reason'][:150]}")
