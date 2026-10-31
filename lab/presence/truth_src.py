"""Truth for the stair and shaft matrix, by eye, before any presence model call. Coordinates are pixels on data/lab/presence/sheets/<b>/<i>.jpg (2000 px long side, raw, not aligned).
Writes lab/presence/truth.json and data/lab/presence/truth_meta.json. No API.

Per object: kind stair|elevator|shaft; cls main|ext|minor; boxes {floor: box} for floors where it is drawn; unsure [floors] left out of scoring.
A floor neither in boxes nor in unsure is an absent cell. A stair means anything with treads. Absence is "as drawn": no symbol on that floor.
Stair boxes marked earlier were carried from lab/audit/truth.json and lab/defects/seeds.py through truth_draft.py and rechecked on galleries; the rest were read for this experiment.
held_out and plant are fixed here, before any model output exists, so they cannot follow results."""
import hashlib
import json
import time
from pathlib import Path

import cv2

ROOT = Path("/Users/tsharma/Github/Meta")
D = json.load(open(ROOT / "data/lab/presence/draft.json"))


def old(b, o, floors, **k):
    return dict(kind=D[b][o]["kind"], cls=k.pop("cls", "main"), boxes={str(f): D[b][o]["boxes"][str(f)] for f in floors}, **k)


def new(kind, cls, boxes, **k):
    return dict(kind=kind, cls=cls, boxes={str(f): v for f, v in boxes.items()}, **k)


def s1(box, *floors, kind="stair", cls="minor"):
    return new(kind, cls, {f: box for f in floors})


T = {}
T["bme_f"] = dict(split="dev", ref=1, discharge=1, objects=dict(
    A_topleft=old("bme_f", "A_topleft", [0, 1, 2]), B_topright=old("bme_f", "B_topright", [0, 1, 2]),
    III=old("bme_f", "III", [0, 1, 2, 3]), II=old("bme_f", "II", [0, 1, 2, 3]), I=old("bme_f", "I", [0, 1, 2, 3]),
    f0_S=s1([948, 1115, 1012, 1168], 0), f0_midE=s1([1385, 553, 1422, 602], 0), f0_NEs=s1([1446, 510, 1484, 540], 0), f0_Ns=s1([563, 303, 592, 330], 0),
    f0_chute=s1([808, 295, 832, 378], 0), f0_SW1=s1([385, 993, 425, 1017], 0), f0_SW2=s1([163, 995, 222, 1032], 0), f0_SW3=s1([60, 1183, 122, 1217], 0),
    f0_farNE=s1([1712, 233, 1790, 262], 0), f0_farE=s1([1748, 520, 1792, 537], 0),
    f1_gal=s1([870, 368, 892, 422], 1), f1_gal2=s1([1003, 517, 1052, 535], 1), f1_lab=s1([1233, 352, 1256, 402], 1), f1_lab2=s1([1372, 518, 1398, 537], 1),
    f1_kon=s1([545, 487, 598, 503], 1), f1_1320=s1([1325, 1095, 1342, 1127], 1), f2_muse=s1([1335, 547, 1352, 592], 2),
    f3_a=s1([408, 552, 422, 568], 3), f3_b=s1([1158, 730, 1176, 802], 3), f3_c=s1([612, 783, 640, 802], 3),
    x_S=s1([848, 1063, 862, 1100], 1, kind="shaft", cls="main"), x_E=s1([1206, 786, 1224, 803], 1, kind="shaft", cls="main"),
    x_f0a=s1([1350, 458, 1370, 482], 0, kind="shaft", cls="main"), x_f0b=s1([1706, 452, 1724, 478], 0, kind="shaft", cls="main"), akna=s1([1240, 482, 1272, 510], 0, kind="shaft", cls="main")))

c = "habs_stelizabeths_center"
T[c] = dict(split="dev", ref=1, discharge=1, objects=dict(
    central=old(c, "central", range(5)), north=old(c, "north", [1, 2]), west_curve=old(c, "west_curve", range(5)), east_curve=old(c, "east_curve", range(5)),
    ext_N=s1([982, 88, 1030, 165], 1, cls="ext"), ext_S=s1([982, 1200, 1020, 1282], 1, cls="ext"), ext_SW=s1([310, 966, 356, 1006], 1, cls="ext"), ext_SE=s1([1660, 966, 1706, 1006], 1, cls="ext"),
    attic_N=s1([1048, 1110, 1070, 1182], 4), attic_C=s1([903, 845, 934, 861], 4),
    elevator=s1([832, 845, 889, 882], 0, 1, 2, 3, 4, kind="elevator", cls="main"),  # floor 0 is labelled EMR/SHAFT
    xshaft_w=old(c, "xshaft_w", [1, 2]),  # seeds.py also listed floor 3; a 6x crop shows a shower with no X there
    xshaft_e=old(c, "xshaft_e", [1, 2, 3, 4]), xshaft_c=old(c, "xshaft_c", [0, 1, 2, 3]),
    xshaft_r=s1([1309, 695, 1324, 711], 0, 1, 2, 3, kind="shaft", cls="main"), xshaft_nw2=s1([340, 463, 365, 479], 0, 1, 2, 3, 4, kind="shaft", cls="main"),
    xshaft_ne2=s1([1651, 460, 1669, 475], 1, 2, 3, 4, kind="shaft", cls="main"), shaft_f0a=s1([1382, 848, 1404, 866], 0, kind="shaft", cls="main"), shaft_f0b=s1([1410, 848, 1437, 866], 0, kind="shaft", cls="main")))

T["nibs_clinic"] = dict(split="dev", ref=0, discharge=0, objects=dict(
    north=new("stair", "main", {0: [790, 188, 830, 263], 1: [730, 319, 764, 396]}),
    central=new("stair", "main", {0: [1045, 363, 1080, 425], 1: [982, 490, 1018, 558]}),
    south=new("stair", "main", {0: [1059, 729, 1127, 761], 1: [986, 860, 1064, 896]}),
    ext_W=s1([598, 738, 618, 762], 0, cls="ext"),
    elevator=new("elevator", "main", {0: [1048, 545, 1078, 570], 1: [984, 674, 1018, 701]})))

T["mml_capitol"] = dict(split="dev", ref=1, discharge=0, objects=dict(
    hall=old("mml_capitol", "hall", [1, 2]),
    rear=new("stair", "ext", {0: [1388, 130, 1560, 322], 1: [1452, 130, 1567, 320], 2: D["mml_capitol"]["rear"]["boxes"]["2"]}, unsure=[2]),  # drawn faded below the upper floor
    front=new("stair", "ext", {0: [1405, 822, 1560, 963], 1: [1415, 827, 1567, 963], 2: D["mml_capitol"]["front"]["boxes"]["2"]}, unsure=[2])))

T["bme_k"] = dict(split="heldout", ref=1, discharge=1, objects=dict(
    **{o: old("bme_k", o, range(4)) for o in ("N_center", "W_grand", "E_grand", "W_wing", "E_wing", "spiral_NE", "curve_SW")},
    **{o: old("bme_k", o, [0, 1]) for o in ("svc_NW", "svc_SW", "svc_NE", "svc_SE")},
    vest_W=new("stair", "minor", {0: [845, 885, 862, 970], 1: [828, 895, 856, 985]}, unsure=[3]), vest_E=new("stair", "minor", {0: [1105, 905, 1125, 985], 1: [1115, 900, 1140, 985]}, unsure=[3]),
    ext_N=s1([1150, 266, 1178, 300], 0, cls="ext"), spiral_E3=s1([1705, 537, 1730, 560], 3),
    lift_E=new("elevator", "main", {0: [1137, 763, 1186, 794], 1: [1117, 774, 1163, 801], 2: [1122, 785, 1172, 810], 3: [1155, 782, 1205, 810]}),  # labelled FELVONO
    xshaft_NW=new("shaft", "main", {0: [843, 510, 870, 535], 2: [838, 530, 862, 556], 3: [864, 530, 890, 555]}),  # X box; floor 1 draws two plain rooms there
    xshaft_NE=new("shaft", "main", {0: [1126, 510, 1155, 534], 2: [1115, 532, 1140, 556], 3: [1148, 530, 1172, 553]})))

v = "habs_va_bldg1"
T[v] = dict(split="heldout", ref=1, discharge=0, objects=dict(
    west=old(v, "west", range(5)), east=old(v, "east", range(4)), central=old(v, "central", range(5)),
    entry_S=new("stair", "ext", {1: [949, 827, 1097, 885], 0: [961, 802, 1112, 835]}, unsure=[0]),  # floor 0 is a ramp with steps
    svc_f0=s1([1225, 545, 1262, 595], 0), steps_f4=s1([823, 672, 846, 697], 4),
    elev1=new("elevator", "main", {0: [972, 535, 1011, 563], 1: [958, 574, 997, 601], 2: [947, 612, 988, 635], 3: [979, 614, 1018, 643], 4: [968, 602, 1010, 634]}),
    elev2=new("elevator", "main", {0: [972, 572, 1011, 595], 1: [958, 613, 997, 635], 2: [947, 645, 988, 668], 3: [979, 649, 1018, 671], 4: [968, 640, 1010, 665]}),
    x_morgue=s1([1102, 394, 1122, 414], 0, kind="shaft", cls="main")))

w = "wfu_huffman"
T[w] = dict(split="heldout", ref=1, discharge=0, objects=dict(
    central=old(w, "central", range(4)), ext_E=old(w, "ext_E", [0, 1], cls="ext"), south=old(w, "south", [2, 3], cls="ext"),  # dashed on floors 2 and 3; off the drawn plan on 0 and 1
    south2=new("stair", "ext", {2: [565, 1700, 660, 1800], 3: [575, 1715, 670, 1800]}, unsure=[2, 3]),
    W_steps=s1([400, 965, 445, 1040], 1, cls="ext"),
    x_SW=new("shaft", "main", {2: [535, 1355, 580, 1395], 3: [545, 1360, 595, 1415]})))

s = "schenley"
T[s] = dict(split="heldout", ref=1, discharge=1, objects=dict(
    sw=old(s, "sw", range(4)), se=old(s, "se", range(4)), n=old(s, "n", range(4)),
    aud_nw=old(s, "aud_nw", [2, 3]), aud_ne=old(s, "aud_ne", [2, 3]), aud_sw=old(s, "aud_sw", [2, 3]), aud_se=old(s, "aud_se", [2, 3]),
    ext_Wd=s1([125, 770, 165, 802], 1, cls="ext"), ext_Ed=s1([1825, 770, 1862, 802], 1, cls="ext"),
    diag_NW=new("stair", "minor", {0: [830, 300, 960, 430]}, unsure=[0]), diag_NE=new("stair", "minor", {0: [1040, 310, 1190, 430]}, unsure=[0]),  # treads or seating, unclear
    vent_w=old(s, "vent_w", [1, 2, 3]), vent_e=old(s, "vent_e", [1, 2, 3]),
    vent_c=new("shaft", "main", {0: [967, 332, 1006, 371], 1: [973, 258, 1014, 297], 2: [978, 329, 1019, 368], 3: [971, 312, 1011, 348]})))

T["habs_stelizabeths_atkins"] = dict(split="heldout", ref=1, discharge=1, objects=dict(
    main=new("stair", "main", {0: [730, 565, 835, 632], 1: [612, 565, 836, 632], 2: [612, 566, 836, 632], 3: [612, 566, 836, 632], 4: [612, 568, 725, 632]}),
    base_DN=s1([905, 535, 950, 605], 0),
    ext_W=s1([250, 568, 332, 632], 1, cls="ext"), ext_NW=s1([425, 335, 482, 402], 1, cls="ext"), ext_NE=s1([1230, 340, 1296, 415], 1, cls="ext"), porch=s1([1220, 938, 1308, 1007], 1, cls="ext"),
    ext_E=new("stair", "ext", {1: [1685, 563, 1736, 637], 2: [1706, 563, 1736, 640], 3: [1708, 562, 1734, 638]}),
    ext_S=new("stair", "ext", {0: [510, 990, 692, 1055], 1: [510, 990, 692, 1058]}),
    x_toilet=new("shaft", "main", {2: [690, 818, 732, 885], 3: [682, 820, 730, 870]})))

# Planted cases, chosen from truth alone. Dev cases exist only to tune; held-out cases are read once.
PLANT = {
    "bme_f": [("middle", "II", 2), ("discharge", "III", 1), ("top", "I", 3)],
    "habs_stelizabeths_center": [("middle", "west_curve", 3), ("discharge", "east_curve", 1), ("top", "central", 4)],
    "bme_k": [("middle", "W_grand", 2), ("discharge", "E_wing", 1), ("top", "N_center", 3)],
    "habs_va_bldg1": [("middle", "west", 2), ("discharge", "central", 0), ("top", "east", 3)],
    "wfu_huffman": [("middle", "central", 2), ("discharge", "central", 0), ("top", "central", 3)],
    "schenley": [("middle", "sw", 2), ("discharge", "se", 1), ("top", "n", 3)],
    "habs_stelizabeths_atkins": [("middle", "main", 2), ("discharge", "main", 1), ("top", "main", 4)],
}
idx = json.load(open(ROOT / "data/lab/presence/sheets/index.json"))
for b, t in T.items():
    n = len(idx[b])
    assert 0 <= t["ref"] < n and 0 <= t["discharge"] < n, b
    for o, v_ in t["objects"].items():
        assert v_["kind"] in ("stair", "elevator", "shaft") and v_["cls"] in ("main", "ext", "minor"), (b, o)
        for f, bx in v_["boxes"].items():
            h, w_ = idx[b][int(f)]["shape"]
            assert 0 <= bx[0] < bx[2] <= w_ and 0 <= bx[1] < bx[3] <= h, (b, o, f, bx, w_, h)
for b, cs in PLANT.items():
    for kind, o, f in cs:
        ob = T[b]["objects"][o]
        assert str(f) in ob["boxes"] and ob["cls"] == "main" and ob["kind"] == "stair", (b, kind, o, f)
        fl = sorted(int(x) for x in ob["boxes"])
        if kind == "middle":
            assert fl[0] < f < fl[-1] and f != T[b]["discharge"], (b, o, f, fl)
        if kind == "top":
            assert f == fl[-1], (b, o, f, fl)
        if kind == "discharge":
            assert f == T[b]["discharge"], (b, o, f)
P = ROOT / "lab/presence/truth.json"
if __name__ == "__main__":
    P.write_text(json.dumps(dict(_note="See lab/presence/truth_src.py. One labeller, not a licensed reviewer.", plant=PLANT, buildings=T), indent=1))
    meta = dict(sha256=hashlib.sha256(P.read_bytes()).hexdigest(), written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(P.stat().st_mtime)), epoch=P.stat().st_mtime)
    (ROOT / "data/lab/presence/truth_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)
    k = {}
    for b, t in T.items():
        for o, v_ in t["objects"].items():
            n = len(idx[b])
            un = set(map(int, v_.get("unsure", [])))
            pres = [int(f) for f in v_["boxes"] if int(f) not in un]
            key = (t["split"], "stair" if v_["kind"] == "stair" else "shaft/lift", v_["cls"])
            r = k.setdefault(key, [0, 0, 0, 0])
            r[0] += 1; r[1] += len(pres); r[2] += n - len(pres) - len(un); r[3] += len(un)
    for key, r in sorted(k.items()):
        print(key, "objects", r[0], "present cells", r[1], "absent cells", r[2], "unsure", r[3])
