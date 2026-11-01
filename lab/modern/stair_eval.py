"""Vector-only stair detection scored against the ground truth boxes in track.py. No API call. Run: .venv/bin/python -I lab/modern/stair_eval.py"""
import json, sys
sys.path.insert(0, "lab/modern")
OUT = "data/lab/modern"
P = json.load(open(OUT + "/prep.json"))
GT = {
    "mml_capitol": {"rear": dict(ground=[1942, 166, 2021, 346], main=[1942, 170, 2022, 350], upper=[1942, 172, 2021, 352]), "front": dict(ground=[1942, 1138, 2021, 1254], main=[1942, 1133, 2022, 1249], upper=[1942, 1130, 2021, 1246]), "interior": dict(main=[1744, 859, 1807, 985], upper=[1744, 748, 1807, 952]),
                    "front turn": dict(main=[1827, 1066, 1942, 1121], upper=[1825, 1066, 1940, 1119])},
    "mml_baxter": {"left": dict(main=[1791, 713, 1854, 839], upper=[1792, 603, 1854, 807]), "right": dict(main=[1864, 713, 1928, 839], upper=[1865, 603, 1928, 807])},
    "nibs_clinic": {"1CS3": {"1": [1196, 286, 1249, 399], "2": [1194, 280, 1250, 400]}, "1AS1": {"1": [1581, 548, 1631, 644], "2": [1579, 547, 1634, 645]}, "1BS2": {"1": [1604, 1104, 1707, 1159], "2": [1582, 1103, 1702, 1153]}},
    "wfu_efird": {"central": {"0": [469, 503, 508, 546], "1": [472, 520, 506, 568], "2": [469, 488, 499, 531], "3": [478, 542, 505, 583]}},
}


def hit(a, b):
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


rows = {}
for id, objs in GT.items():
    tp = fn = fp = 0
    for lv in P[id]["levels"]:
        runs = [r["box_reg"] for r in P[id]["runs"][lv]]
        gts = [o[lv] for o in objs.values() if lv in o]
        for g in gts:
            if any(hit(r, g) for r in runs):
                tp += 1
            else:
                fn += 1
        fp += sum(not any(hit(r, g) for g in gts) for r in runs)
    rows[id] = dict(tp=tp, fn=fn, fp=fp)
    print(id, rows[id])
json.dump(rows, open(OUT + "/stair_eval.json", "w"), indent=1)
