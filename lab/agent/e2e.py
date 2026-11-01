"""End-to-end agent run on the clinic, scored against the frozen e2e_truth.json.

Run: env LAB_NAME=agent LAB_BUDGET=5 LAB_TOTAL=75 .venv/bin/python lab/agent/e2e.py <spark|wonder> <effort> <run> [usd_cap] [max_turns]
"""
import collections, json, re, sys
from pathlib import Path
sys.path.insert(0, "lab/agent")
import fixture as F  # before loop: fixture sets the path order for lab/modern's vec
import loop

D = Path("data/lab/agent/e2e"); D.mkdir(exist_ok=True)
MODELS = {"spark": "muse-spark-1.3", "wonder": "muse-wonder-1.0"}
NUM = re.compile(r"\d+(?:\.\d+)?")


def score(rec):
    tools = rec["tools"]
    txt = " ".join(json.dumps(t["result"]) if not isinstance(t["result"], str) else t["result"] for t in tools if t["name"] != "file_finding")
    known = set(NUM.findall(txt))
    ids = {t["call_id"] for t in tools}
    finds = [t["args"] for t in tools if t["name"] == "file_finding" and isinstance(t["args"], dict)]
    rows = []
    for f in finds:
        blob = " ".join(str(f.get(k, "")) for k in ("sheet", "location", "summary", "printed", "measured"))
        low = blob.lower()
        rows.append(dict(kind=f.get("kind"), sheet=f.get("sheet"), summary=f.get("summary"), printed=f.get("printed"), measured=f.get("measured"), bad_ids=[c for c in f.get("evidence_call_ids", []) if c not in ids], no_evidence=not f.get("evidence_call_ids"),
                         numbers_not_in_tool_text=sorted({n for n in NUM.findall(blob) if n not in known}),
                         seed_A="south" in low or "2bs2" in low, seed_B="central" in low or "2as1" in low, seed_C="2875" in low or "2663" in low, service="service" in low,
                         mispair=[r for r in ("1500", "1030", "2420", "1763", "2910") if r in low and f.get("kind") == "dimension_mismatch"]))
    views = [t for t in tools if t["name"] == "view_sheet"]
    return dict(model=rec["model"], effort=rec["effort"], stop=rec["stop"], turns=len(rec["turns"]), usd=round(rec["usd"], 4), seconds=rec["seconds"], tool_counts=dict(collections.Counter(t["name"] for t in tools)),
                tool_errors=sum(isinstance(t["result"], str) and t["result"].startswith("tool error") for t in tools), overlay_views=sum(isinstance(t["args"], dict) and bool(t["args"].get("overlay_row")) for t in views),
                viewed_059=any(isinstance(t["args"], dict) and str(t["args"].get("sheet")) == "059" for t in views), findings=rows,
                filed_A=any(r["seed_A"] for r in rows), filed_C=any(r["seed_C"] for r in rows), filed_B_false=[r["summary"] for r in rows if r["seed_B"]], filed_service_false=[r["summary"] for r in rows if r["service"]], filed_mispairs=[r["mispair"] for r in rows if r["mispair"]],
                prose_after=rec["text"][-1][:300] if rec["text"] else "", prose_chars=sum(len(t) for t in rec["text"]), cached_share=round(sum(t["cached"] for t in rec["turns"]) / max(sum(t["input"] for t in rec["turns"]), 1), 3), incomplete=rec.get("incomplete", 0))


if __name__ == "__main__":
    key, effort, run = sys.argv[1], sys.argv[2], sys.argv[3]
    cap = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    turns = int(sys.argv[5]) if len(sys.argv) > 5 else 14
    p = D / f"{key}_{effort}_{run}.json"
    if p.exists():
        rec = json.load(open(p))
    else:
        t = F.Tools()
        filed = []

        def file_finding(a):
            filed.append(a)
            return {"filed": len(filed)}
        h = {n: getattr(t, n) for n in ("list_sheets", "view_sheet", "get_stair_matrix", "measure_route", "read_table")}
        h["file_finding"] = file_finding
        # run 0 of Spark had no wrap-up warning and ended with nothing filed; every later run has it, so those runs are post hoc on the loop, not on the truth
        rec = loop.run(MODELS[key], F.SYSTEM, F.USER, F.TOOLS, h, effort=effort, max_turns=turns, max_output_tokens=6000, usd_cap=cap, wrap_up=3)
        json.dump(rec, open(p, "w"), indent=1, default=str)
    s = score(rec)
    json.dump(s, open(D / f"{key}_{effort}_{run}.score.json", "w"), indent=1)
    print(json.dumps(s, indent=1))
    print("agent spent", round(loop.m.spent("agent"), 4))
