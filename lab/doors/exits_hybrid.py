"""Clinic exits: vector doors near the outside region, each cropped and judged by Spark. Key: IFC IsExternal, minus fence gates."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab"); sys.path.insert(0, str(Path(__file__).parent))
import fitz
import metalab as m
import key as K
from exits_hybrid_lib import cluster
from vector_eval import quarter, in_box, footprint
from spark_tiles import CACHE, score

SCH = {"type": "object", "additionalProperties": False, "required": ["leads_outside", "confidence"],
       "properties": {"leads_outside": {"type": "boolean"}, "confidence": {"type": "number"}}}
PROMPT = ("This is a crop of a building floor plan. The red circle marks one door. Does this door connect the inside of the building directly to the outdoors "
          "(an exterior door or exit door)? Answer false if both sides of the door are rooms, corridors, stairs or any other indoor space. "
          "Outdoors is the blank area outside the thick outer wall. Reply with leads_outside and a confidence from 0 to 1.")


def judge(tag, i, mid):
    f = CACHE / f"crop2_{tag}_{int(mid[0])}_{int(mid[1])}.json"
    if f.exists():
        return json.loads(f.read_text())
    R = 7 * K.PT_PER_M
    r = fitz.Rect(mid[0] - R, mid[1] - R, mid[0] + R, mid[1] + R)
    dpi = int(72 * 900 / (2 * R))
    im = K.render(tag, dpi, clip=r)
    cv2.circle(im, (im.shape[1] // 2, im.shape[0] // 2), 38, (0, 0, 255), 3)
    o, info = m.spark(PROMPT, images=[im], schema=SCH, effort="low")
    out = {**o, "mid": mid, "usd": info["usd"]}
    f.write_text(json.dumps(out))
    return out


res = {}
for tag in K.SHEETS:
    A, L, arcs, reg = K.clinic_key(tag)
    door = cluster([a for a in arcs if a["leaf"] or quarter(a)], K.PT_PER_M)
    z = 110 / 72
    wall = K.wall_ink(tag, z)
    for a in door:
        cv2.line(wall, (int(a["cx"] * z), int(a["cy"] * z)), (int(a["shut_end"][0] * z), int(a["shut_end"][1] * z)), 1, 5)
    lab, oid = K.outside(1 - wall)
    out = lab == oid
    near = cv2.dilate(out.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(5 * K.PT_PER_M * z) | 1,) * 2)).astype(bool)
    mids = [[(a["cx"] + a["shut_end"][0]) / 2, (a["cy"] + a["shut_end"][1]) / 2] for a in door]
    cand = [p for p in mids if near[int(p[1] * z), int(p[0] * z)]]
    T = {}
    for l in L:
        if l["external"] and "Chain Link" not in l["name"]:
            T.setdefault(l["door"], []).append([(l["s_hinge"][0] + l["s_shut"][0]) / 2, (l["s_hinge"][1] + l["s_shut"][1]) / 2])
    T = [np.mean(v, axis=0).tolist() for v in T.values()]
    with ThreadPoolExecutor(6) as ex:
        js = list(ex.map(lambda it: judge(tag, it[0], it[1]), enumerate(cand)))
    yes = [j["mid"] for j in js if j["leads_outside"]]
    # double doors give two yes points: merge within 1.5 m
    from spark_tiles import nms
    yes_m = nms(yes, 1.5 * K.PT_PER_M)
    tp = score(yes_m, T, 1.5 * K.PT_PER_M)
    cand_tp = score(nms(cand, 1.5 * K.PT_PER_M), T, 1.5 * K.PT_PER_M)
    res[tag] = {"doors": len(door), "candidates": len(cand), "truth_external_no_gates": len(T), "candidates_cover_truth": cand_tp, "spark_yes": len(yes), "yes_merged": len(yes_m), "tp": tp, "usd": sum(j["usd"] for j in js)}
    print(tag, res[tag], flush=True)
    b = np.array(T)
    v = K.render(tag, 110)
    for p in T:
        cv2.circle(v, (int(p[0] * z), int(p[1] * z)), 16, (0, 170, 0), 2)
    for p in yes_m:
        cv2.drawMarker(v, (int(p[0] * z), int(p[1] * z)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
    cv2.imwrite(str(K.OUT / f"exits_hybrid_{tag}.png"), v[200:1500, 1300:3000])
(K.OUT / "exits_hybrid.json").write_text(json.dumps(res, indent=1))
