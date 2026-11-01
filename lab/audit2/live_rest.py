"""L5, L6, L7 from data/lab/audit2/rule.json. Usage: live_rest.py agent | concept | exits. Writes only to data/lab/audit2."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "lab"); sys.path.insert(0, "lab/agent")
import metalab as m
AUD = Path("data/lab/audit2")


def agent():
    import prompts as P
    D = Path("data/lab/agent")
    cases = json.load(open(D / "cases.json")); T = json.load(open(D / "truth.json"))["cases"]
    def one(a):
        c, model = a
        out, info = m.spark(c["claim"], images=[str(D / c["image"])], model=model, schema=P.SCHEMA, effort="low", system=P.SYSTEM, max_tokens=8000)
        return c["id"], out
    with ThreadPoolExecutor(4) as ex:
        sp = dict(ex.map(one, [(c, "muse-spark-1.3") for c in cases]))
        wo = dict(ex.map(one, [(c, "muse-wonder-1.0") for c in cases if T[c["id"]]["cat"] == "route"]))
    json.dump(dict(spark=sp, wonder=wo), open(AUD / "L5_agent.json", "w"), indent=1)
    for name, R, cfg in (("spark low", sp, "spark_low"), ("wonder low routes", wo, "wonder_low")):
        old = {i: json.load(open(D / f"runs/{cfg}/{i}.json"))["out"]["verdict"] for i in R}
        flips = [(T[i]["key"], T[i]["label"], old[i], R[i]["verdict"]) for i in R if old[i] != R[i]["verdict"]]
        bad = [i for i in R if T[i]["label"] == "bad"]; good = [i for i in R if T[i]["label"] == "good"]
        cats = {c: (sum(R[i]["verdict"] == "defect" for i in bad if T[i]["cat"] == c), sum(T[i]["cat"] == c for i in bad), sum(R[i]["verdict"] == "defect" for i in good if T[i]["cat"] == c), sum(T[i]["cat"] == c for i in good)) for c in ("stair", "route", "door", "value")}
        print(name, "repeat caught", sum(R[i]["verdict"] == "defect" for i in bad), "/", len(bad), "false rej", sum(R[i]["verdict"] == "defect" for i in good), "/", len(good), "flips", len(flips), flips, cats)
    print("spent", round(m.spent("audit2"), 4))


def concept():
    S = Path("data/lab/samtricks/sheets")
    print(sorted(p.name for p in S.iterdir())[:8])
    out = {}
    for k in ("n161", "n52", "n133"):
        p = next(S.glob(f"{k}.*"))
        rows = []
        for r in range(3):
            q = m.sam_image("floor plan", str(p))
            rows.append(sorted(round(float(x.confidence), 3) for x in q if x.confidence is not None))
        out[k] = rows
        print(k, rows, flush=True)
    json.dump(out, open(AUD / "L6_concept.json", "w"), indent=1)
    print("spent", round(m.spent("audit2"), 4))


def exits():
    assert m.spent("audit2") < 2.4
    sys.path.insert(0, "lab/declared")
    import read_spark as RS
    SID = "nibs_clinic"
    res = {}
    for k, want in (("c161", 1440), ("c162", 294)):
        s = RS.CL[k]
        clips = RS.tiles(SID, s)
        def one(a):
            i, c, run = a
            r, info = m.spark(RS.EXIT_P, images=[RS.render(SID, s, 2000, c)], schema=RS.EXIT_S, effort="medium", max_tokens=6000)
            return i, run, r
        with ThreadPoolExecutor(4) as ex:
            ans = list(ex.map(one, [(i, c, run) for run in (0, 1) for i, c in enumerate(clips)]))
        for run in (0, 1):
            kept, raw = [], 0
            for i, rr, r in ans:
                if rr != run:
                    continue
                c = clips[i]
                for e in r["exits"]:
                    x, y = c[0] + e["x"] / 1000 * (c[2] - c[0]), c[1] + e["y"] / 1000 * (c[3] - c[1])
                    raw += e["capacity"] or 0
                    if e["capacity"] and (x < .5) == (c[0] < .25) and (y < .5) == (c[1] < .25):
                        kept.append((round(x * 1000), round(y * 1000), e["width"], e["capacity"]))
            res[f"{k}_{run}"] = dict(sum=sum(q[3] for q in kept), want=want, raw_sum=raw, kept=kept)
            print(k, "read", run, "half-sheet sum", sum(q[3] for q in kept), "want", want, "raw", raw, "kept", kept, flush=True)
    json.dump(res, open(AUD / "L7_exits.json", "w"), indent=1)
    print("spent", round(m.spent("audit2"), 4))


{"agent": agent, "concept": concept, "exits": exits}[sys.argv[1]]()
