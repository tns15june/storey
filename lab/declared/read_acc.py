"""Reading accuracy with no model call: Spark reads against the hand truth and the clinic file names. Writes data/lab/declared/read_acc.json."""
import json, re, sys
sys.path.insert(0, "lab/declared")
import read_spark as RS
import rules as R
import posthoc as P
import run as RN
OUT = P.OUT
t = json.load(open(OUT / "truth.json"))
res = {}
# clinic title blocks against the NIBS file names, an independent transcription
tb = RS.load("tbc/")
same, diff = 0, []
for f in RS.clinic_files():
    r = tb.get((f"tbc/{f}", 0))
    name = re.sub(r"^Clinic_\d+_", "", f[:-4])
    if r and R.title_same(name, r["sheet_title"]):
        same += 1
    elif r:
        diff.append((f[7:10], name[:44], r["sheet_title"][:44]))
res["clinic_title_vs_filename"] = dict(n=len(RS.clinic_files()), same=same, diff=diff)
res["clinic_title_block_numbers_reread"] = dict(second_reads=sum(k[1] == 1 for k in tb), disagreed=[k[0][4:18] for k in tb if k[1] == 1 and R.norm(tb[(k[0], 0)]["sheet_number"]) != R.norm(tb[k]["sheet_number"])])
res["sequence_field_misread"] = sum(1 for k, v in tb.items() if k[1] == 0 and v["sequence"] != str(int(k[0][11:14])) and "_036_" not in k[0] and "_293_" not in k[0])
# code fields, both reads, against hand truth
want = {"usc_leconte/1": dict(occupancy="B", construction="IIB", sprinklered="no", occupant_load="43", area="4,351"),
        "carpinteria_adu1/0": dict(occupancy="R-3", construction="VB", area="400"),
        "carpinteria_adu2/0": dict(occupancy="R-3", construction="VB", area="480"),
        "bcoe_addendum3/1": dict(occupancy="B", sprinklered="yes", occupant_load="357", area="9,273"),
        "mml_capitol/0": dict(area="1,728"), "mml_baxter/0": dict(area="2,304"),
        f"nibs_clinic/{RS.CL['c161']}": dict(occupancy="BUSINESS", construction="II", sprinklered="yes"), f"nibs_clinic/{RS.CL['c162']}": dict(occupancy="BUSINESS", construction="II", sprinklered="yes")}
n = ok = 0
bad = []
for s, w in want.items():
    for run in (0, 1):
        r = RS._done[(f"code/{s}", run)]["out"]
        for k, v in w.items():
            n += 1
            hit = v.lower() in str(r[k]).lower()
            ok += hit
            if not hit:
                bad.append((s[:24], run, k, r[k][:40]))
res["code_fields_two_reads"] = dict(n=n, ok=ok, bad=bad)
print(json.dumps(res, indent=0)[:3000])
(OUT / "read_acc.json").write_text(json.dumps(res, indent=1))

# USC reads against the hand truth written before any model call
u = t["usc_leconte"]
a, b = RN.merged("index/usc", "rows")
idx = [sum(x["sheet_number"] in u["index"] and R.norm(x["title"]) == R.norm(u["index"][x["sheet_number"]]) for x in q) for q in (a, b)]
a, b = RN.merged("doors/usc", "rows")
drow = [sum(x["door"] in u["doors"] and x["hardware"].replace(" ", "") == u["doors"][x["door"]][1].replace(" ", "") and (x["fire_rating"] or "-") == u["doors"][x["door"]][2] for x in q) for q in (a, b)]
a, b = RN.merged("tags/usc", "tags")
tags = [len({x["tag"] for x in q} & set(u["tags_on_a101"])) for q in (a, b)]
ghost = [sorted({x["tag"] for x in q} - set(u["tags_on_a101"])) for q in (a, b)]
a, b = RN.merged("hw/usc", "sets")
hw = [sum(sorted(s["doors"]) == sorted(u["hardware_sets"][s["set"].replace(" ", "")]) for s in q) for q in (a, b)]
a, b = RN.merged("finish/usc", "rows")
fin = [sum(R.norm(x["room_name"]) == R.norm(u["finish_rooms"].get(x["room_no"], "")) for x in q) for q in (a, b)]
tbk = sum(RS._done[(f"tb/usc/{p}", r)]["out"]["sheet_number"] == v[0] for p, v in u["title_blocks_pages_11_18"].items() for r in (0, 1))
res["usc_vs_hand_truth"] = dict(index=(idx, 19), doors=(drow, 21), tags=(tags, 21), tag_ghosts=ghost, hardware_sets=(hw, 4), finish=(fin, 24), title_block_numbers=(tbk, 16))
tb2 = RS.load("tbc/")
ix = {x["sheet_number"]: x["title"] for x in RN.merged("index/clinic", "rows")[0]}
for n in ("242", "244"):
    k = next(k for k in tb2 if f"Clinic_{n}_" in k[0])
    res[f"clinic_{n}"] = (tb2[k], ix.get(tb2[k]["sheet_number"]))
print(json.dumps({k: res[k] for k in ("usc_vs_hand_truth", "clinic_242", "clinic_244")})[:900])
(OUT / "read_acc.json").write_text(json.dumps(res, indent=1))
