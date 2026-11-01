"""Independent rescore of the agent test bed from cached answers. No API, imports nothing from lab/agent. Run with -I."""
import json, math
from pathlib import Path
D = Path("data/lab/agent")
T = json.load(open(D / "truth.json"))
print({k: (str(v)[:160]) for k, v in T.items() if k.startswith("_") and k != "_rule"})
C = T["cases"]
def load(cfg):
    return {p.stem: json.load(open(p)) for p in (D / "runs" / cfg).glob("case_*.json")}
def flag(r): return bool(r and r.get("out") and r["out"]["verdict"] == "defect")
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 2), round(c + h, 2)
for cfg in ("spark_low", "spark_medium", "wonder_low", "wonder_medium", "spark_low_blind"):
    R = load(cfg)
    bad = [i for i in R if C[i]["label"] == "bad"]; good = [i for i in R if C[i]["label"] == "good"]
    k, f = sum(flag(R[i]) for i in bad), sum(flag(R[i]) for i in good)
    real = [i for i in bad if C[i]["source"] == "real"]
    cats = {c: (sum(flag(R[i]) for i in bad if C[i]["cat"] == c), sum(C[i]["cat"] == c for i in bad), sum(flag(R[i]) for i in good if C[i]["cat"] == c), sum(C[i]["cat"] == c for i in good)) for c in ("stair", "route", "door", "value")}
    print(cfg, f"caught {k}/{len(bad)} ci {wilson(k, len(bad))} false_rej {f}/{len(good)} ci {wilson(f, len(good))} real {sum(flag(R[i]) for i in real)}/{len(real)} usd/call {sum(r.get('usd', 0) for r in R.values()) / len(R):.4f}", cats)
    print("   wrong passes", [C[i]["key"] for i in bad if not flag(R[i])], "| false rej", [C[i]["key"] for i in good if flag(R[i])])
print("source mix", {(c["label"], c["cat"], c["source"]) for c in C.values()})
import collections
print(collections.Counter((c["label"], c["cat"], c["source"]) for c in C.values()))
print("D04 related cases", [(i, c["key"], c["label"], c["cat"]) for i, c in C.items() if "D04" in c["key"] or "d04" in c["key"].lower()])
cases = {c["id"]: c for c in json.load(open(D / "cases.json"))}
print(json.dumps(json.load(open(D / "e2e_truth.json")), indent=0)[:1700])
