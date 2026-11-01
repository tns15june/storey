"""Roll up every result into final.json. No API calls."""
import sys, glob; sys.path.insert(0, "lab/seed")
from common import *
from union import cluster, evaluate, CONF
import score_sam

ALL = ["bme_f", "schenley", "habs_stelizabeths_atkins", "habs_stelizabeths_center", "mml_capitol", "nibs_clinic", "mml_baxter"]
out = {"buildings": {}}
tot = {"sam": [0, 0], "spark": [0, 0], "fusion_or": [0, 0]}
floors_tot = {"present_n": 0, "present_sam": 0, "present_spark": 0, "absent_n": 0, "absent_sam": 0, "absent_spark": 0}
for bid in ALL:
    nf = len(CAT[bid]["floors"])
    cl = cluster(bid, "doc")
    tr = json.load(open(OUT / "sam" / f"union_doc__{bid}.json"))["tracks"]
    g = json.load(open(OUT / "sam" / f"gate_doc_{bid}.json"))
    keep_all = [True] * len(cl)
    rows, false = evaluate(bid, cl, tr, keep_all)
    rr = json.load(open(OUT / "sam" / f"rebox_result__{bid}.json"))
    keep_gate = [g.get(str(i), {}).get("is_stair", True) for i in range(len(cl))]
    rg, fg = evaluate(bid, cl, tr, keep_gate)
    keep_soft = [g.get(str(i), {}).get("is_stair", True) or len(c["floors"]) >= 2 for i, c in enumerate(cl)]
    rs, fs = evaluate(bid, cl, tr, keep_soft)
    main = {k: v for k, v in rows.items() if v["cls"] == "main"}
    stair = {k: v for k, v in rows.items() if v["cls"] in ("main", "ext")}
    # three readings of each main stack: SAM alone, Spark alone, and present if fusion_or says so
    n = len(main)
    sam_ok = sum(v["all_ok"] for v in main.values())
    spark_ok = sum(v["seeded"] and all(x == "ok" for x in v["spark"].values()) for v in main.values())
    fusion_or_ok = 0
    for k, v in stair.items():
        t = next(x for x in TRUTH[bid] if x["id"] == k)
        if not v["seeded"]:
            continue
        for f in range(nf):
            if f in t.get("any", []):
                continue
            sam, sp = v["floors"][f], v["spark"][f]
            if f in t["floors"]:
                floors_tot["present_n"] += 1; floors_tot["present_sam"] += sam == "ok"; floors_tot["present_spark"] += sp == "ok"
            else:
                floors_tot["absent_n"] += 1; floors_tot["absent_sam"] += sam == "ok"; floors_tot["absent_spark"] += sp == "ok"
    for k, v in main.items():
        t = next(x for x in TRUTH[bid] if x["id"] == k)
        if v["seeded"]:
            # present if SAM (conf >= 0.5) or Spark fires: a present floor needs one right, an absent floor needs both silent
            fusion_or_ok += all(f in t.get("any", []) or ((v["floors"][f] == "ok" or v["spark"][f] == "ok") if f in t["floors"] else (v["floors"][f] == "ok" and v["spark"][f] == "ok")) for f in range(nf))
    tot["sam"][0] += sam_ok; tot["sam"][1] += n
    tot["spark"][0] += spark_ok; tot["spark"][1] += n
    tot["fusion_or"][0] += fusion_or_ok; tot["fusion_or"][1] += n
    rm = {k: v for k, v in rr["rebox"]["rows"].items() if v["cls"] == "main"}
    cnt = lambda rows_, c: [sum(x["cls"] == c for x in rows_.values()), sum(x["cls"] == c and x["seeded"] for x in rows_.values()), sum(x["cls"] == c and x["all_ok"] for x in rows_.values())]
    out["buildings"][bid] = {
        "floors": nf, "clusters": len(cl), "main_n": n, "main_seeded": sum(v["seeded"] for v in main.values()),
        "main_e2e_sam_conf05": sam_ok, "main_e2e_spark_only": spark_ok, "main_e2e_fusion_or": fusion_or_ok,
        "ext": cnt(rows, "ext"), "minor": cnt(rows, "minor"), "false_clusters": len(false), "dups": sum(v.get("dups", 0) for v in rows.values()),
        "gate_hard": {"kept": sum(keep_gate), "false": len(fg), "main_seeded": sum(rg[k]["seeded"] for k in main), "dropped_main": [k for k in main if not rg[k]["seeded"]]},
        "gate_soft": {"kept": sum(keep_soft), "false": len(fs), "main_seeded": sum(rs[k]["seeded"] for k in main)},
        "seed_only_doc": {c: cnt(rr["doc"]["rows"], c) for c in ("main", "ext", "minor")}, "seed_only_rebox": {c: cnt(rr["rebox"]["rows"], c) for c in ("main", "ext", "minor")},
        "false_doc": len(rr["doc"]["false"]), "false_rebox": len(rr["rebox"]["false"]), "rebox_bad_main": {k: v.get("floors") for k, v in rm.items() if not v["all_ok"]},
        "align": {k: v["overlap"] for k, v in json.load(open(OUT / bid / "align.json")).items()},
    }
    b = out["buildings"][bid]
    print(f"{bid:26s} fl {nf} clusters {len(cl):2d} main {n} seeded {b['main_seeded']} e2e sam {sam_ok} spark {spark_ok} fusion_or {fusion_or_ok} | ext {b['ext']} minor {b['minor']} false {len(false)} dups {b['dups']} | hardgate kept {b['gate_hard']['kept']} false {b['gate_hard']['false']} droppedmain {b['gate_hard']['dropped_main']} | softgate kept {b['gate_soft']['kept']} false {b['gate_soft']['false']} | rebox main {b['seed_only_rebox']['main']} minor {b['seed_only_rebox']['minor']} false {b['false_rebox']} (doc minor {b['seed_only_doc']['minor']} false {b['false_doc']}) bad {b['rebox_bad_main']}")
out["totals"] = tot
out["floor_pairs"] = floors_tot
sumk = lambda key, c: [sum(b[key][c][i] for b in out["buildings"].values()) for i in range(3)]
out["seed_only"] = {name: {c: sumk(name, c) for c in ("main", "ext", "minor")} for name in ("seed_only_doc", "seed_only_rebox")}
out["false"] = {"doc": sum(b["false_doc"] for b in out["buildings"].values()), "rebox": sum(b["false_rebox"] for b in out["buildings"].values()),
                "hard_gate": sum(b["gate_hard"]["false"] for b in out["buildings"].values()), "soft_gate": sum(b["gate_soft"]["false"] for b in out["buildings"].values()),
                "clusters_raw": sum(b["clusters"] for b in out["buildings"].values()), "clusters_hard": sum(b["gate_hard"]["kept"] for b in out["buildings"].values()), "clusters_soft": sum(b["gate_soft"]["kept"] for b in out["buildings"].values())}
out["shift_test"] = json.load(open(OUT / "shift_test.json"))
led = [json.loads(x) for x in open(ROOT / "data/lab/ledger.jsonl")]
led = [x for x in led if x["name"] == "seed"]
out["usd"] = round(sum(x["usd"] for x in led), 3)
out["calls"] = {"spark": sum(x["model"].startswith("muse") for x in led), "sam_image": sum(x.get("kind") == "image" for x in led), "sam_video": sum(x.get("kind") == "video" for x in led)}
out["sam_video_seconds_median"] = float(np.median([x["seconds"] for x in led if x.get("kind") == "video"]))
print(json.dumps({k: out[k] for k in ("totals", "floor_pairs", "seed_only", "false", "usd", "calls", "sam_video_seconds_median")}, indent=0))
json.dump(out, open(OUT / "final.json", "w"), indent=1, default=int)
