"""Run the frozen overlay cases through one config and score it. Each answer is cached, so a rerun spends nothing.

Run: env LAB_NAME=agent LAB_BUDGET=5 LAB_TOTAL=75 .venv/bin/python lab/agent/audit.py <model-key> <effort> [n_cases] [blind]
"""
import json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "lab"); sys.path.insert(0, "lab/agent")
import metalab as m
import prompts as P

D = Path("data/lab/agent")
MODELS = {"spark": "muse-spark-1.3", "wonder": "muse-wonder-1.0"}
BAD_BY = ("cat", "source")


def one(a):
    cfg, model, effort, case, blind = a
    p = D / "runs" / cfg / f"{case['id']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return json.load(open(p))
    t0 = time.time()
    try:
        out, info = m.spark(case["claim"] + (P.BLIND if blind else ""), images=[] if blind else [str(D / case["image"])], model=model, schema=P.SCHEMA, effort=effort, system=P.SYSTEM, max_tokens=8000)
        row = dict(id=case["id"], out=out, usd=info["usd"], seconds=info["seconds"], input=info["input"], output=info["output"])
    except (json.JSONDecodeError, TypeError) as e:  # truncated answer: paid for, counted apart, not retried
        row = dict(id=case["id"], out=None, error=repr(e)[:200], usd=0.0, seconds=time.time() - t0)
    json.dump(row, open(p, "w"), indent=1)
    return row


def score(cfg, rows, truth):
    T = truth["cases"]
    bad = [r for r in rows if T[r["id"]]["label"] == "bad"]
    good = [r for r in rows if T[r["id"]]["label"] == "good"]
    flag = lambda r: bool(r["out"]) and r["out"]["verdict"] == "defect"
    s = dict(cfg=cfg, n=len(rows), caught=sum(map(flag, bad)), n_bad=len(bad), false_rej=sum(map(flag, good)), n_good=len(good), unparsed=sum(r["out"] is None for r in rows),
             usd=round(sum(r["usd"] for r in rows), 4), usd_per=round(sum(r["usd"] for r in rows) / max(len(rows), 1), 4), sec_med=round(sorted(r["seconds"] for r in rows)[len(rows) // 2], 1))
    s["caught_real"] = sum(flag(r) for r in bad if T[r["id"]]["source"] == "real"); s["n_real"] = sum(T[r["id"]]["source"] == "real" for r in bad)
    s["by_cat"] = {c: [sum(flag(r) for r in bad if T[r["id"]]["cat"] == c), sum(T[r["id"]]["cat"] == c for r in bad), sum(flag(r) for r in good if T[r["id"]]["cat"] == c), sum(T[r["id"]]["cat"] == c for r in good)] for c in ("stair", "route", "door", "value")}
    loc = [r for r in bad if T[r["id"]]["cat"] == "route" and flag(r)]
    hit = lambda r: any(math.dist((r["out"]["x"], r["out"]["y"]), d[:2]) <= 150 for d in T[r["id"]]["defect"])
    s["route_located"] = sum(map(hit, loc))
    s["wrong_passes"] = [T[r["id"]]["key"] for r in bad if not flag(r)]
    s["false_rejections"] = [T[r["id"]]["key"] for r in good if flag(r)]
    return s


if __name__ == "__main__":
    key, effort = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    blind = len(sys.argv) > 4
    cases = json.load(open(D / "cases.json"))[:n]
    truth = json.load(open(D / "truth.json"))
    cfg = f"{key}_{effort}" + ("_blind" if blind else "")
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, [(cfg, MODELS[key], effort, c, blind) for c in cases]))
    s = score(cfg, rows, truth)
    json.dump(s, open(D / "runs" / f"{cfg}.json", "w"), indent=1)
    print(json.dumps(s))
    print("agent spent", round(m.spent("agent"), 4), "ledger", round(m.spent(), 3))
