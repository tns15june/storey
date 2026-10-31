"""POST HOC, dev only: why seeds miss and two ways to fix it. P1 SAM text concept on a zoomed crop; P2 Spark on a zoomed crop asking for the whole stair, two rounds. Also centre error of Spark's one-call list."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import dev1, seeds as sd

WHOLE_P = ('You are an object grounding expert. This is a zoomed crop of an architectural floor plan. Find the staircase nearest the centre of the crop and give one box tight around the whole stair symbol: '
           'all of its flights of tread lines and the landings between them, out to the stair walls. Do not include corridors, the rooms beside it, door swings or labels. If there is no staircase, return no object.' + TAIL)


def crop_img(b, fr, k, minside, long=1000):
    x1, y1, x2, y2 = sd.crop_window(b, fr.shape, k=k, minside=minside)
    c = fr[y1:y2, x1:x2]
    z = min(5.0, long / max(c.shape[:2]))
    return cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA), (x1, y1, x2, y2), z


def sam_concept(a, words=("staircase",)):
    """SAM text concept on a zoomed crop. Returns candidate boxes with confidence, in stack px. Cached."""
    st, key, fr, b = a
    p = SPK / f"{st}_samc{key}.json"
    if p.exists():
        return json.load(open(p))
    c, (x1, y1, x2, y2), z = crop_img(b, fr, 3.0, 300)
    out = []
    for w in words:
        for r in m.sam_image(w, c):
            q = r.bounds
            out.append(dict(word=w, box=[q.left / z + x1, q.top / z + y1, q.right / z + x1, q.bottom / z + y1], conf=None if r.confidence is None else float(r.confidence)))
    json.dump(out, open(p, "w"))
    return out


def spark_whole(a):
    """Two rounds: crop on the box, ask, re-crop tighter on the answer, ask again. Cached per round."""
    st, key, fr, b = a
    cur = b
    for rnd, (k, ms) in enumerate(((3.0, 300), (1.8, 200))):
        p = SPK / f"{st}_whole{key}_r{rnd}.json"
        if p.exists():
            o = json.load(open(p))
        else:
            c, (x1, y1, x2, y2), z = crop_img(cur, fr, k, ms)
            q = R / "crops" / f"{st}_whole{key}_r{rnd}.png"; cv2.imwrite(str(q), c)
            r, info = m.spark(WHOLE_P, images=[str(q)], schema=SCHEMA)
            h, w = y2 - y1, x2 - x1
            o = dict(boxes=[[v + d for v, d in zip(to_px(t, w, h), (x1, y1, x1, y1))] for t in r["objects"]], info=info)
            json.dump(o, open(p, "w"))
        if not o["boxes"]:
            return dict(rounds=rnd, box=cur)
        # nearest to the current centre when Spark lists more than one
        cx, cy = (cur[0] + cur[2]) / 2, (cur[1] + cur[3]) / 2
        cur = min(o["boxes"], key=lambda t: np.hypot((t[0] + t[2]) / 2 - cx, (t[1] + t[3]) / 2 - cy))
        if rnd == 0:
            first = cur
    return dict(rounds=2, box=cur, first=first)


if __name__ == "__main__":
    print("spark one-call centre error is on held-out only, printed elsewhere")
    res = {"loose": [], "concept": [], "whole1": [], "whole2": []}
    for st in DEV:
        cl = dev1.loose_clusters(st); fr = frames(st)
        jobs = []
        for i, c in enumerate(cl):
            own = [k for k in assign(st, c["seed_box"]) if TRUTH[st]["stairs"][k]["cls"] != "amb"]
            if own:
                jobs.append((st, str(i + 1), fr[c["seed_floor"]], c["seed_box"], own, c["seed_floor"]))
        with ThreadPoolExecutor(6) as ex:
            con = list(ex.map(sam_concept, [j[:4] for j in jobs]))
            who = list(ex.map(spark_whole, [j[:4] for j in jobs]))
        for j, cc, ww in zip(jobs, con, who):
            t = TRUTH[st]["stairs"][j[4][0]]["boxes"].get(str(j[5]))
            if not t:
                continue
            cen = ((j[3][0] + j[3][2]) / 2, (j[3][1] + j[3][3]) / 2)
            best = max(cc, key=lambda x: x["conf"] or 0) if cc else None
            res["loose"].append(iou(j[3], t)); res["concept"].append(iou(best["box"], t) if best else 0)
            res["whole1"].append(iou(ww.get("first", ww["box"]), t)); res["whole2"].append(iou(ww["box"], t))
            print(st, j[4], "loose", round(iou(j[3], t), 2), "concept", best and round(iou(best["box"], t), 2), best and round(best["conf"] or 0, 2), "n", len(cc), "whole1", round(iou(ww.get("first", ww["box"]), t), 2), "whole2", round(iou(ww["box"], t), 2), flush=True)
    for k, v in res.items():
        a = np.array(v); print(k, "n", len(a), "median", round(float(np.median(a)), 3), ">=.5", int((a >= .5).sum()))
    print("spent", round(m.spent("tightseed"), 4))
