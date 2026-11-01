"""Score Spark boxes per sheet against truth and draw overlays. No API calls. Usage: score_spark.py variant"""
import sys; sys.path.insert(0, "lab/seed")
from common import *
var = sys.argv[1]
tot = {}
rows = []
for bid, truth in TRUTH.items():
    if bid.startswith("_"):
        continue
    for i in range(len(CAT[bid]["floors"])):
        p = OUT / "spark" / f"{bid}_f{i}_{var}.json"
        if not p.exists():
            continue
        im = cv2.imread(str(OUT / bid / f"f{i}.png"))
        h, w = im.shape[:2]
        boxes = [to_px(o, w, h) for o in json.load(open(p))["objects"]]
        here = [t for t in truth if i in t["floors"] and t["cls"] != "elevator"]
        used, found = set(), {}
        for t in here:
            for k, b in enumerate(boxes):
                if k not in used and hit(b, t["box"]):
                    used.add(k); found[t["id"]] = b; break
        # a box on a stair absent from this sheet's floor list, or a second box on the same stair, is not a new false stair
        extra = [k for k in range(len(boxes)) if k not in used]
        dup = [k for k in extra if any(hit(boxes[k], t["box"]) for t in here)]
        false = [k for k in extra if k not in dup]
        for t in truth:
            c = (0, 170, 0) if t["id"] in found else (0, 0, 255) if t in here else (200, 200, 200)
            if t in here:
                cv2.rectangle(im, tuple(t["box"][:2]), tuple(t["box"][2:]), c, 2)
                cv2.putText(im, t["id"], (t["box"][0], t["box"][1] - 3), 0, 0.5, c, 1)
        for k, b in enumerate(boxes):
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (255, 0, 255) if k in false else (255, 120, 0), 2)
            if k in false:
                cv2.putText(im, f"F{k}", (b[0], b[3] + 14), 0, 0.5, (255, 0, 255), 1)
        cv2.imwrite(str(OUT / "spark" / f"{bid}_f{i}_{var}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        by = lambda c: f"{sum(t['id'] in found for t in here if t['cls']==c)}/{sum(t['cls']==c for t in here)}"
        print(f"{bid:28s} f{i} boxes {len(boxes):2d} main {by('main')} ext {by('ext')} minor {by('minor')} false {len(false)} dup {len(dup)} missed {[t['id'] for t in here if t['id'] not in found]}")
        rows.append({"bid": bid, "floor": i, "boxes": len(boxes), "found": list(found), "here": [t["id"] for t in here], "false": len(false), "dup": len(dup)})
json.dump(rows, open(OUT / f"spark_score_{var}.json", "w"), indent=1)
sel = lambda c: [(t["id"] in r["found"]) for r in rows for t in TRUTH[r["bid"]] if t["id"] in r["here"] and t["cls"] == c]
for c in ("main", "ext", "minor"):
    s = sel(c); print(var, c, sum(s), "of", len(s))
print(var, "false boxes", sum(r["false"] for r in rows), "of", sum(r["boxes"] for r in rows))
