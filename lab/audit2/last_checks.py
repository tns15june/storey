"""Remaining no-API checks. Run with -I."""
import glob, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/audit2")
import rescore_tightseed as S
R = S.R
# Spark verdict veto on loose and snap
def vetoed(me):
    rows = []
    for st in S.HELD:
        tr = json.load(open(R / "sam" / f"{st}__{me}.json"))
        tr2 = json.loads(json.dumps(tr))
        n = 0
        for o, fl in tr2["tracks"].items():
            for f, mk in list(fl.items()):
                p = R / "spark" / f"{st}_chk{me}_{o}_{f}.json"
                if p.exists() and not json.load(open(p))["stair"]:
                    mk["area"] = 0; n += 1
        rows += S.rows_of(st, tr2)
    return S.tally(rows)
for me in ("loose_960", "snap_960"):
    print("Spark verdict veto", me, vetoed(me))
# Round 1 misalignment and seed centre error
A = json.load(open("/Users/tsharma/Github/Meta/lab/audit/truth.json"))
for b, k in (("wfu_huffman", "central"), ("habs_va_bldg1", "west"), ("habs_va_bldg1", "central")):
    c = np.array([[(x[0] + x[2]) / 2, (x[1] + x[3]) / 2] for x in A[b][k]["boxes"].values()]) if isinstance(A[b][k], dict) and "boxes" in A[b][k] else None
    if c is not None:
        print("Round 1 frame spread", b, k, "x", round(float(np.ptp(c[:, 0]))), "y", round(float(np.ptp(c[:, 1]))))
errs = {}
for st in S.HELD + ["bme_f", "habs_stelizabeths_center", "mml_capitol"]:
    for p in glob.glob(str(R / "spark" / f"{st}_f[0-9].json")):
        f = p.rsplit("_f", 1)[1].split(".")[0]
        for b in json.load(open(p))["boxes"]:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            best = None
            for s in S.ALL[st]["stairs"].values():
                t = s["boxes"].get(f)
                if t:
                    d = np.hypot(cx - (t[0] + t[2]) / 2, cy - (t[1] + t[3]) / 2)
                    if best is None or d < best[0]:
                        best = (d, cx - (t[0] + t[2]) / 2, cy - (t[1] + t[3]) / 2)
            if best and best[0] <= 130:
                errs.setdefault(st, []).append(best[1:])
print("Spark box centre error, median abs px", {k: (round(float(np.median(np.abs(np.array(v)[:, 0])))), round(float(np.median(np.abs(np.array(v)[:, 1])))), len(v)) for k, v in errs.items()}, "n", sum(len(v) for v in errs.values()))
# presence gate count, order, cost
O = "/Users/tsharma/Github/Meta/data/lab/presence/"
n = p = 0; low = []
for f in glob.glob(O + "align/*.json"):
    d = json.load(open(f)); b = os.path.basename(f)[:-5]
    for k, v in d.items():
        if v.get("is_ref"):
            continue
        n += 1; p += v["aligned"]
        if v["inliers"]: low.append((v["inliers"], b, k))
print("gate floors", n, "pass", p, "lowest", sorted(low)[:3])
sp = json.load(open(O + "spark_sheets.json"))
IDX = json.load(open(O + "sheets/index.json"))
J = [b for b in IDX if f"joint/{b}" in sp]
bad = [b for b in J if [sp[f"joint/{b}"]["shown"][q - 1] for q in sp[f"joint/{b}"]["read"]["bottom_to_top"]] != list(range(len(IDX[b])))]
print("joint orders", len(J))
print("buildings whose joint order differs from index order:", bad)
cost = {}
T = json.load(open("/Users/tsharma/Github/Meta/lab/presence/truth.json"))["buildings"]
for b, t in T.items():
    if t["split"] == "heldout":
        E = json.load(open(O + f"cells/{b}.json"))
        sh = sum(v["info"]["usd"] for k, v in sp.items() if k.startswith(b + "/") or k == f"joint/{b}")
        cost[b] = (round(E["compare_usd"], 3), round(E["crop_usd"], 3), round(sh, 3), E["crops"])
print("cost per held-out building compare, crops, sheets, ncrops", cost, "mean", round(np.mean([sum(v[:3]) for v in cost.values()]), 3))
# declared frozen code hashes
st = json.load(open("/Users/tsharma/Github/Meta/data/lab/declared/stamps.json"))["v2_frozen_before_held_out"]
for k, v in st.items():
    if isinstance(v, dict):
        print(k, "same" if hashlib.sha256(open("/Users/tsharma/Github/Meta/" + k, "rb").read()).hexdigest() == v["sha256"] else "CHANGED")
