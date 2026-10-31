"""No-human pipeline: Spark boxes on every sheet, cluster across floors, seed SAM once per cluster, Spark crop gate, score.
Usage: union.py <variant> [buildings]. Cached by sam_track.run and the gate file."""
import sys; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab")
from common import *
from sam_track import frames, video, spark_boxes, pad, run, REPEAT
from concurrent.futures import ThreadPoolExecutor
import metalab as m

CONF = 0.5
GATE = {"type": "object", "additionalProperties": False, "required": ["results"], "properties": {"results": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["image", "what", "is_stair"],
    "properties": {"image": {"type": "integer"}, "what": {"type": "string"}, "is_stair": {"type": "boolean"}}}}}}


def linked(a, b):
    area = lambda x: max(1, (x[2] - x[0]) * (x[3] - x[1]))
    c = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    small, big = (a, b) if area(a) < area(b) else (b, a)
    cs = c[0] if small is a else c[1]
    inside = big[0] <= cs[0] <= big[2] and big[1] <= cs[1] <= big[3]
    return iou(a, b) >= 0.15 or (inside and area(big) < 5 * area(small))


def cluster(bid, var, cls=None):
    items = [(f, b) for f in range(len(CAT[bid]["floors"])) for b in spark_boxes(bid, f, var, cls)]
    par = list(range(len(items)))
    find = lambda i: i if par[i] == i else find(par[i])
    for i, j in ((i, j) for i in range(len(items)) for j in range(i)):
        if linked(items[i][1], items[j][1]):
            par[find(i)] = find(j)
    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(items[i])
    out = []
    for g in groups.values():
        med = max(g, key=lambda x: sum(iou(x[1], y[1]) for y in g))  # the box most members agree with survives one bad box
        out.append({"seed_floor": med[0], "seed_box": med[1], "members": g, "floors": sorted({f for f, _ in g})})
    return sorted(out, key=lambda c: -len(c["floors"]))


def crop(bid, c):
    f, b = c["seed_floor"], c["seed_box"]
    im = frames(bid)[f]
    h, w = im.shape[:2]
    mx, my = max(40, (b[2] - b[0]) // 2), max(40, (b[3] - b[1]) // 2)
    x1, y1, x2, y2 = max(0, b[0] - mx), max(0, b[1] - my), min(w, b[2] + mx), min(h, b[3] + my)
    s = 512 / max(x2 - x1, y2 - y1)
    out = cv2.resize(im[y1:y2, x1:x2], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(out, (round((b[0] - x1) * s), round((b[1] - y1) * s)), (round((b[2] - x1) * s), round((b[3] - y1) * s)), (0, 0, 255), 2)
    return out


def gate(bid, var, cl):
    path = OUT / "sam" / f"gate_{var}_{bid}.json"
    if path.exists():
        return json.load(open(path))
    res = {}
    for s in range(0, len(cl), 10):
        part = cl[s:s + 10]
        prompt = (f"These {len(part)} images are crops of an architectural floor plan, numbered 1 to {len(part)} in order. Each has a red box. "
                  "For each image say what the drawing shows inside the red box and whether a staircase (a run of closely spaced parallel treads) is there. "
                  "Hatching, wall poche, radiators, lockers, seating, dimension marks and text are not stairs.")
        r, info = m.spark(prompt, images=[crop(bid, c) for c in part], schema=GATE)
        for x in r["results"]:
            res[str(s + x["image"] - 1)] = {"is_stair": x["is_stair"], "what": x["what"]}
    json.dump(res, open(path, "w"), indent=1)
    return res


def evaluate(bid, cl, tracks, keep):
    """Per truth stair: was it seeded, and is every floor right using SAM masks at conf >= CONF."""
    truth = TRUTH[bid]
    nf = len(CAT[bid]["floors"])
    rows = {}
    claimed = set()
    for t in truth:
        cands = [i for i, c in enumerate(cl) if keep[i] and any(f in t["floors"] and hit(b, t["box"], 0.25) for f, b in c["members"])]
        if not cands:
            rows[t["id"]] = {"cls": t["cls"], "seeded": False, "all_ok": False}
            continue
        claimed.update(cands)

        def floors_of(i):
            tr = tracks.get(str(i + 1), {})
            out, spark_out = {}, {}
            for f in range(nf):
                mk = tr.get(str(f))
                has = bool(mk and mk.get("area", 0) > 0 and (mk.get("conf") or 0) >= CONF)
                if f in t.get("any", []):
                    out[f] = spark_out[f] = "ok"
                    continue
                if f in t["floors"]:
                    out[f] = "miss" if not has else ("ok" if hit(mk["box"], t["box"], 0.25) else "drift")
                else:
                    out[f] = "ghost" if has else "ok"
                sp = f in cl[i]["floors"]
                spark_out[f] = "ok" if sp == (f in t["floors"]) else ("spark_miss" if not sp else "spark_ghost")
            return out, spark_out

        # shortcut: a stair is credited to its best cluster, duplicates are counted separately, a product must merge them
        i = max(cands, key=lambda i: (sum(v == "ok" for v in floors_of(i)[0].values()), sum(hit(b, t["box"], 0.25) for _, b in cl[i]["members"])))
        out, spark_out = floors_of(i)
        rows[t["id"]] = {"cls": t["cls"], "seeded": True, "cluster": i, "seed_floor": cl[i]["seed_floor"], "floors": out, "spark": spark_out, "all_ok": all(v == "ok" for v in out.values()), "dups": len(cands) - 1}
    false = [i for i, c in enumerate(cl) if keep[i] and i not in claimed and not any(hit(c["seed_box"], t["box"], 0.25) for t in truth)]
    return rows, false


def go(bid, var):
    cl = cluster(bid, var)
    wh = video(bid)[1]
    seeds = [(i + 1, c["seed_floor"], c["seed_box"]) for i, c in enumerate(cl)]
    r = run(f"union_{var}__{bid}", bid, seeds)
    g = gate(bid, var, cl)
    keep_all = [True] * len(cl)
    keep = [g.get(str(i), {}).get("is_stair", True) for i in range(len(cl))]
    summary = {}
    for name, k in (("raw", keep_all), ("gated", keep)):
        rows, false = evaluate(bid, cl, r["tracks"], k)
        summary[name] = {"rows": rows, "false_clusters": false, "clusters": sum(k)}
        mn = [x for x in rows.values() if x["cls"] == "main"]
        ex = [x for x in rows.values() if x["cls"] == "ext"]
        bad = {kk: {f: v for f, v in x["floors"].items() if v != "ok"} for kk, x in rows.items() if x["cls"] == "main" and x["seeded"] and not x["all_ok"]}
        print(f"{bid:26s} {var:8s} {name:5s} clusters {sum(k):2d} false {len(false):2d} main seeded {sum(x['seeded'] for x in mn)}/{len(mn)} e2e {sum(x['all_ok'] for x in mn)}/{len(mn)} ext e2e {sum(x['all_ok'] for x in ex)}/{len(ex)} bad {bad} missed {[kk for kk, x in rows.items() if x['cls'] == 'main' and not x['seeded']]}")
    json.dump({"clusters": cl, "gate": g, "summary": summary}, open(OUT / "sam" / f"result_{var}__{bid}.json", "w"), indent=1, default=int)


if __name__ == "__main__":
    var = sys.argv[1]
    bids = sys.argv[2:] or ["bme_f", "schenley", "habs_stelizabeths_atkins", "habs_stelizabeths_center"]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda b: go(b, var), bids))
