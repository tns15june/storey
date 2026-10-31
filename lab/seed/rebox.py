"""Does zoomed Spark re-grounding fix offset boxes on small stairs? Crop each cluster's seed, ask Spark again at 4x, re-track. Usage: rebox.py [buildings]"""
import sys; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab")
from common import *
from union import cluster, CONF, evaluate
from sam_track import frames, run
from seed_spark import SCHEMA, DOC
from concurrent.futures import ThreadPoolExecutor
import metalab as m

BIDS = sys.argv[1:] or ["bme_f", "schenley", "habs_stelizabeths_atkins", "habs_stelizabeths_center", "mml_capitol", "nibs_clinic", "mml_baxter"]


def one(BID, cl, i):
    path = OUT / "sam" / f"rebox_{BID}_{i}.json"
    if path.exists():
        return json.load(open(path))
    im_all = frames(BID)
    h, w = im_all[0].shape[:2]
    c = cl[i]
    b = c["seed_box"]
    cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
    r = max(120, (max(b[2] - b[0], b[3] - b[1]) * 3) // 2)
    x1, y1, x2, y2 = max(0, cx - r), max(0, cy - r), min(w, cx + r), min(h, cy + r)
    crop = im_all[c["seed_floor"]][y1:y2, x1:x2]
    s = 1024 / max(crop.shape[:2])
    big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    res, info = m.spark(DOC, images=[big], schema=SCHEMA, effort="low")
    ch, cw = big.shape[:2]
    best, bd = None, 1e9
    for o in res["objects"]:  # the box nearest the crop centre is the one this crop was made for
        p = to_px(o, cw, ch)
        pb = [x1 + p[0] / s, y1 + p[1] / s, x1 + p[2] / s, y1 + p[3] / s]
        d = np.hypot((pb[0] + pb[2]) / 2 - cx, (pb[1] + pb[3]) / 2 - cy)
        if d < bd:
            best, bd = [round(v) for v in pb], d
    out = {"box": best, "n": len(res["objects"]), "shift": None if best is None else round(float(bd), 1), "usd": info["usd"]}
    json.dump(out, open(path, "w"))
    return out


def go(BID):
    cl = cluster(BID, "doc")
    with ThreadPoolExecutor(3) as ex:
        rb = list(ex.map(lambda i: one(BID, cl, i), range(len(cl))))
    print(BID, "calls", len(rb), "usd", round(sum(r["usd"] for r in rb), 4), "no stair in crop", sum(r["box"] is None for r in rb), "median shift px", np.median([r["shift"] for r in rb if r["box"]]))
    kept = [i for i, r in enumerate(rb) if r["box"]]
    cl2 = [dict(cl[i], seed_box=rb[i]["box"], orig=i) for i in kept]
    r2 = run(f"union_rebox__{BID}", BID, [(j + 1, c["seed_floor"], c["seed_box"]) for j, c in enumerate(cl2)])
    base = json.load(open(OUT / "sam" / f"union_doc__{BID}.json"))
    out = {}
    for name, c, tr in (("doc", cl, base["tracks"]), ("rebox", cl2, r2["tracks"])):
        # a stair counts only if a cluster's seed box itself lands on it, so an offset box gets no credit from its neighbours
        only = [dict(x, members=[(x["seed_floor"], x["seed_box"])]) for x in c]
        rows, false = evaluate(BID, only, tr, [True] * len(only))
        out[name] = {"rows": rows, "false": false, "clusters": len(c)}
        print(BID, name, {k: [sum(x["cls"] == k for x in rows.values()), sum(x["cls"] == k and x["seeded"] for x in rows.values()), sum(x["cls"] == k and x["all_ok"] for x in rows.values())] for k in ("main", "ext", "minor")}, "clusters", len(c), "false", len(false),
              "bad main", {k: x.get("floors") for k, x in rows.items() if x["cls"] == "main" and not x["all_ok"]})
    json.dump(out, open(OUT / "sam" / f"rebox_result__{BID}.json", "w"), indent=1, default=int)


with ThreadPoolExecutor(4) as ex:
    list(ex.map(go, BIDS))
