"""Score SAM tracks against truth. No API calls. Writes sam_score.json and prints one row per run."""
import sys, glob; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed")
from common import *


def outcome(t, tr, nf):
    """Per-floor outcome of one track `tr` against truth stair `t`."""
    res = {}
    for f in range(nf):
        m = (tr or {}).get(str(f))
        has = bool(m and m.get("area", 0) > 0)
        if f in t["floors"]:
            res[f] = "miss" if not has else ("ok" if hit(m["box"], t["box"], 0.25) else "drift")
        else:
            res[f] = "ok" if not has else "ghost"
    return res


def score(run):
    bid = run["bid"]
    truth = TRUTH[bid]
    nf = len(CAT[bid]["floors"])
    assign, false = {}, []
    for oid, f, b in run["seeds"]:
        cands = [t for t in truth if f in t["floors"] and hit(b, t["box"], 0.25)]
        if not cands:
            false.append(oid); continue
        t = max(cands, key=lambda t: iou(b, t["box"]))
        if t["id"] not in assign or iou(b, t["box"]) > assign[t["id"]][1]:
            assign[t["id"]] = (oid, iou(b, t["box"]), f)
    rows = {}
    for t in truth:
        if t["id"] in assign:
            oid, _, f = assign[t["id"]]
            o = outcome(t, run["tracks"].get(str(oid)), nf)
            rows[t["id"]] = {"cls": t["cls"], "seeded": True, "seed_floor": f, "floors": o, "all_ok": all(v == "ok" for v in o.values())}
        else:
            rows[t["id"]] = {"cls": t["cls"], "seeded": False, "all_ok": False}
    return rows, false


if __name__ == "__main__":
    allr = {}
    for p in sorted(glob.glob(str(OUT / "sam" / "*__f*__*.json"))):  # the seed-floor matrix runs only
        tag = Path(p).stem
        run = json.load(open(p))
        rows, false = score(run)
        allr[tag] = {"rows": rows, "false_seeds": false, "n_seeds": len(run["seeds"]), "false_tracks": [o for o in false if len(run["tracks"].get(str(o), {})) > 1]}
        main = [r for r in rows.values() if r["cls"] == "main"]
        pairs = [v for r in main if r["seeded"] for f, v in r["floors"].items() if int(f) != r["seed_floor"]]
        bad = {k: {f: v for f, v in r["floors"].items() if v != "ok"} for k, r in rows.items() if r["cls"] == "main" and r["seeded"] and not r["all_ok"]}
        print(f"{tag:48s} seeds {len(run['seeds']):2d} false {len(false):2d} | main seeded {sum(r['seeded'] for r in main)}/{len(main)} e2e {sum(r['all_ok'] for r in main)}/{len(main)} | unprompted pairs ok {pairs.count('ok')}/{len(pairs)} | bad {bad} | notseeded {[k for k, r in rows.items() if r['cls']=='main' and not r['seeded']]}")
    json.dump(allr, open(OUT / "sam_score.json", "w"), indent=1)
