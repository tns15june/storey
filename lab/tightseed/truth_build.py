"""Write truth before any model call on tightseed frames. Audit boxes are warped through H; new stacks are read by eye from zoomed grid views. No API."""
import sys, json, hashlib, time
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import truthview as tv
R = tv.ROOT / "data/lab/tightseed"


def rep(box, floors):
    return {str(f): box for f in floors}


T = {"_note": "One labeller, not a licensed reviewer. Boxes are in tightseed stack pixels. Floors absent from `boxes` and from `exclude` have no stair of that identity. cls main and short are scored; ext is scored separately; amb is never scored and a seed on it is neither right nor false. Written before any model call on these frames in this experiment."}
for name, n, crop in (("bme_k", 4, [0, 0, 2000, 1208]), ("habs_va_bldg1", 5, [280, 210, 1810, 1060]), ("wfu_huffman", 4, [200, 100, 1850, 3300])):
    T[name] = dict(n=n, crop=crop, split="heldout", source="lab/audit/truth.json warped through H.json", stairs=tv.warped(name))

all4 = range(4)
T["queens_royce1"] = dict(n=4, crop=[90, 720, 900, 1470], split="heldout", source="eye, grid views; west block of Phase 1, crop fixed before any model call", stairs={
    "R_A": dict(cls="main", boxes=rep([238, 1043, 273, 1075], all4)),
    "R_B": dict(cls="main", boxes=rep([512, 887, 545, 915], all4)),
    "R_C": dict(cls="main", boxes=rep([760, 1042, 797, 1075], all4)),
    "R_D": dict(cls="main", boxes=rep([388, 1127, 424, 1159], all4)),
    "R_E": dict(cls="main", boxes=rep([135, 1289, 168, 1316], all4)),
    "R_G": dict(cls="short", boxes=rep([680, 1076, 708, 1118], (0, 1)), note="corridor flight, ends above f1; f2 grey shaft, f3 nothing"),
    "R_I": dict(cls="short", boxes={"0": [279, 1122, 303, 1139], "1": [282, 1124, 307, 1140]}, note="short flight, ends above f1; f2 and f3 blank roof"),
    "R_H": dict(cls="amb", boxes=rep([640, 1056, 692, 1076], all4), note="stripes outside the wall above f0, could be sunshade"),
    "S_A": dict(cls="amb", boxes=rep([212, 1035, 226, 1078], (1, 2, 3))),
    "S_C": dict(cls="amb", boxes=rep([810, 1035, 823, 1074], (1, 2, 3))),
    "S_D": dict(cls="amb", boxes=rep([435, 1125, 449, 1172], (1, 2, 3))),
})
T["leonard4"] = dict(n=4, crop=[90, 1130, 1960, 2010], split="heldout", source="eye, grid views; Leonard Hall 1st to 4th floor, ground floor dropped because it never registered (18 inliers, blend view wrong)", stairs={
    "L1": dict(cls="main", boxes={"0": [465, 1345, 502, 1396], **rep([467, 1345, 504, 1396], (1, 2, 3))}),
    "L2": dict(cls="main", boxes=rep([850, 1345, 888, 1396], all4)),
    "L3": dict(cls="main", boxes={"0": [1113, 1314, 1152, 1373], **rep([1125, 1312, 1161, 1371], (1, 2, 3))}),
    "L4": dict(cls="main", boxes={"0": [1500, 1233, 1540, 1291], **rep([1501, 1235, 1537, 1293], (1, 2, 3))}),
    "L5_terrace": dict(cls="ext", boxes={"0": [995, 1852, 1062, 1888]}, note="terrace stair, 1st floor sheet only"),
})
T["mml_abbot"] = dict(n=4, crop=[140, 50, 1700, 1330], split="heldout", source="eye, grid views; panels foundation, main, upper, attic of sheet A2.0", stairs={
    "ext_stair": dict(cls="main", boxes={"1": [1197, 1063, 1476, 1167], "2": [1062, 1062, 1478, 1168]}, exclude=[0], note="f0 foundation shows a dashed footprint, excluded; f3 attic has none"),
})
# dev: thresholds only, never reported as scores
T["bme_f"] = dict(n=4, crop=[0, 0, 2000, 1466], split="dev", source="lab/align/stack_sam.py B", stairs={
    "I": dict(cls="main", boxes=rep([848, 900, 945, 1015], all4)), "II": dict(cls="main", boxes=rep([1175, 660, 1275, 740], all4)), "III": dict(cls="main", boxes=rep([512, 652, 612, 735], all4)),
    "TL": dict(cls="main", boxes=rep([418, 215, 530, 280], (0, 1, 2))), "TR": dict(cls="main", boxes=rep([1275, 228, 1385, 290], (0, 1, 2)))})
T["habs_stelizabeths_center"] = dict(n=5, crop=[0, 0, 2000, 1545], split="dev", source="lab/align/stack_sam.py B plus eye for north", stairs={
    "curvW": dict(cls="main", boxes=rep([308, 861, 398, 951], range(5))), "curvE": dict(cls="main", boxes=rep([1624, 861, 1714, 951], range(5))), "central": dict(cls="main", boxes=rep([865, 766, 945, 806], range(5))),
    "north": dict(cls="main", boxes={"1": [1007, 310, 1040, 365], "2": [1023, 312, 1044, 400]})})
T["mml_capitol"] = dict(n=3, crop=[0, 0, 1120, 2520], split="dev", source="lab/seed/truth.json union boxes, seed frames", stairs={
    "int": dict(cls="main", boxes=rep([300, 1340, 425, 1855], (1, 2))), "rear": dict(cls="main", boxes=rep([560, 250, 840, 745], (0, 1)), exclude=[2]), "front": dict(cls="main", boxes=rep([440, 2000, 840, 2360], (0, 1)), exclude=[2])})

p = R / "truth.json"
if p.exists():
    raise SystemExit("truth.json exists: never rewrite it")
# v2: three amb boxes moved onto their stripes after the truth overlay, before any model call. v1 sha 1ac7ff1d...2361 at 15:51:05.
s = json.dumps(T, indent=1)
p.write_text(s)
h = hashlib.sha256(s.encode()).hexdigest()
(R / "truth.sha256").write_text(f"{h}  truth.json  written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
print(h, time.strftime('%Y-%m-%dT%H:%M:%S%z'))
print({k: (v["split"], len(v["stairs"]), sum(len(s["boxes"]) for s in v["stairs"].values())) for k, v in T.items() if k != "_note"})
