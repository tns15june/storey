"""Score Spark reads against the hand truth in data/lab/declared/truth.json. No model calls."""
import json, re, sys, collections
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/declared")
import read_spark as RS
import rules as R
import run as RN
T = json.load(open("/Users/tsharma/Github/Meta/data/lab/declared/truth.json"))
out = {}
# clinic doors
truth = {x.upper() for k in ("doors_079", "doors_080", "doors_081") for x in T["nibs_clinic"][k]}
a, b = RN.merged("doors/clinic", "rows")
ka = {R.norm(x["door"]).replace(" ", "") for x in a}
kb = {R.norm(x["door"]).replace(" ", "") for x in b}
out["clinic_door_rows"] = dict(truth=len(truth), read0=len(ka), read1=len(kb), miss0=sorted(truth - ka), miss1=sorted(truth - kb), extra0=sorted(ka - truth), extra1=sorted(kb - truth), both=len(ka & kb & truth))
sp = T["nibs_clinic"]["door_specials"]
rows = {R.norm(x["door"]).replace(" ", ""): x for x in a}
rows1 = {R.norm(x["door"]).replace(" ", ""): x for x in b}
ok = [k for k, (w, f) in sp.items() if k in rows and R.norm(w).replace(" ", "") in R.norm(rows[k]["width"]).replace(" ", "") and (not f or f in rows[k]["fire_rating"])]
out["clinic_door_specials"] = dict(n=len(sp), ok=len(ok), bad={k: (rows.get(k) or {}).get("width") for k in sp if k not in ok})
# clinic finish against the model
m = RN.ifc_spaces()
f0, f1 = RN.merged("finish/clinic", "rows")
n0 = {R.norm(x["room_no"]).replace(" ", ""): x["room_name"] for x in f0}
n1 = {R.norm(x["room_no"]).replace(" ", ""): x["room_name"] for x in f1}
out["clinic_finish"] = dict(read0=len(n0), read1=len(n1), both=len(set(n0) & set(n1)), name_disagree={k: (n0[k], n1[k]) for k in set(n0) & set(n1) if R.norm(n0[k]) != R.norm(n1[k])}, only0=sorted(set(n0) - set(n1)), only1=sorted(set(n1) - set(n0)), model=len(m))
# clinic index sample
a, b = RN.merged("index/clinic", "rows")
ia = {x["sheet_number"]: x for x in a}
ib = {x["sheet_number"]: x for x in b}
s = T["nibs_clinic"]["index_sample"]
out["clinic_index"] = dict(read0=len(ia), read1=len(ib), rows0=len(a), rows1=len(b), sample=len(s), sample_ok=sum(k in ia and R.title_same(ia[k]["title"], v) for k, v in s.items()), not_used=[k for k in T["nibs_clinic"]["index_not_used_rows"] if k in ia and k in ib])
# clinic exits
for k, want in (("c161", T["nibs_clinic"]["exit_capacity_sum_161"]), ("c162", T["nibs_clinic"]["exit_capacity_sum_162"])):
    ex = RN.tile_exits(f"exits/clinic/{k}", "nibs_clinic", RS.CL[k])
    e = [[(x["x"], x["y"], x["width"], x["capacity"]) for x in q] for q in ex]
    both, merged = RN.C.dedupe_markers(e)
    out[f"exits_{k}"] = dict(want_sum=want, raw_sum=[sum(x[3] for x in q) for q in e], dedup_sum=[sum(x[3] for x in q) for q in merged], both_sum=sum(x[3] for x in both), with_cap=[sum(1 for x in q if x[3]) for q in merged], total=[len(q) for q in merged], caps=[sorted(x[3] for x in q if x[3]) for q in merged])
# title blocks against Round 1 truth
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/bakeoff")
import truth_meta as tm
import sheets as sh
tb = RS.load("tbc/")
n = g = 0
badtb = []
for key, (cls, nums, _, _) in tm.T.items():
    f = sh.ALL.get(key, (None,))[0]
    if key.startswith("n") and f:
        fn = f.split("/")[-1]
        r = tb.get((f"tbc/{fn}", 0))
        if r:
            n += 1
            okk = r["sheet_number"] in nums
            g += okk
            if not okk:
                badtb.append((key, r["sheet_number"], nums))
out["clinic_title_blocks_vs_round1_truth"] = dict(n=n, ok=g, bad=badtb)
seqs = collections.Counter(v["sequence"] for k, v in tb.items() if k[1] == 0)
out["clinic_tb_sequence_dupes"] = {k: v for k, v in seqs.items() if v > 1}
print(json.dumps(out, indent=1)[:4500])
