"""Shared pieces: frames, crop stacks, SAM tracking with a cache, strict scoring, Spark grounding. All Meta calls go through metalab."""
import warnings; warnings.filterwarnings("ignore")
import hashlib, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import metalab as m

ROOT = Path("/Users/tsharma/Github/Meta"); R = ROOT / "data/lab/tightseed"; ST = R / "stacks"; SAMD = R / "sam"; SPK = R / "spark"
for d in (SAMD, SPK, R / "crops"):
    d.mkdir(exist_ok=True)
TRUTH = json.load(open(R / "truth.json"))
if (R / "truth_fresh.json").exists():
    TRUTH.update({k: v for k, v in json.load(open(R / "truth_fresh.json")).items() if not k.startswith("_")})
STACKS = [k for k in TRUTH if not k.startswith("_")]
DEV = [k for k in STACKS if TRUTH[k]["split"] == "dev"]
HELD = [k for k in STACKS if TRUTH[k]["split"] == "heldout"]
FRESH = [k for k in STACKS if TRUTH[k]["split"] == "fresh"]
SCORED = ("main", "short")
CONF = .5  # Round 1 presence rule


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.0


def hit(box, t, grow=.25):
    """Round 1 rule (lab/seed/common.py), grow .25 as used by heldout_score."""
    gx, gy = (t[2] - t[0]) * grow, (t[3] - t[1]) * grow
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    inside = t[0] - gx <= cx <= t[2] + gx and t[1] - gy <= cy <= t[3] + gy
    big = (box[2] - box[0]) * (box[3] - box[1]) > 4 * (t[2] - t[0]) * (t[3] - t[1])
    return iou(box, t) >= .25 or (inside and not big)


def frames(stack):
    return [cv2.imread(str(ST / stack / f"f{i}.png")) for i in range(TRUTH[stack]["n"])]


def crop_scale(stack, res):
    x1, y1, x2, y2 = TRUTH[stack]["crop"]
    s = min(1.0, res / max(x2 - x1, y2 - y1)) if res else 1.0
    return (x1, y1, x2, y2), s


def to_video(b, crop, s):
    return [int(round((b[0] - crop[0]) * s)), int(round((b[1] - crop[1]) * s)), int(round((b[2] - crop[0]) * s)), int(round((b[3] - crop[1]) * s))]


def mp4(frs, crop, s, path):
    x1, y1, x2, y2 = crop
    w = int((x2 - x1) * s) & ~1; h = int((y2 - y1) * s) & ~1
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "4", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frs:
        p.stdin.write(cv2.resize(f[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_AREA).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return w, h


def track(tag, stack, seeds, res=960, frs=None, direction="both", crop=None, upscale=False):
    """seeds: (object id, floor, box in stack px). One id on several floors is the multi-frame prompt. Returns {id: {floor: {box, area, conf}}} in stack px. Cached by tag."""
    p = SAMD / f"{tag}.json"
    if p.exists():
        return json.load(open(p))
    if crop is None:
        crop, s = crop_scale(stack, res)
    else:
        s = res / max(crop[2] - crop[0], crop[3] - crop[1])
        s = s if upscale else min(1.0, s)
    frs = frs or frames(stack)
    v = SAMD / f"{tag}.mp4"
    w, h = mp4(frs, crop, s, v)
    ids = sorted({o for o, _, _ in seeds})
    tr, errors = {}, []
    masks = {}
    for k in range(0, len(ids), 16):
        part = set(ids[k:k + 16]); byf = {}
        for o, f, b in seeds:
            if o in part:
                x1, y1, x2, y2 = to_video(b, crop, s)
                x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w - 1, max(x2, x1 + 2)), min(h - 1, max(y2, y1 + 2))
                byf.setdefault(f, []).append(f"{o}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}|>")
        prompt = "".join(f"<{f}f>" + ",".join(q) for f, q in sorted(byf.items()))
        for r in m.sam_video(prompt, v, frames=len(frs), direction=direction):
            b = r.bounds; a = int(m.raster(r).sum())
            tr.setdefault(str(int(r.object_id)), {})[str(r.frame.frame_index)] = dict(
                box=[round(b.left / s + crop[0], 1), round(b.top / s + crop[1], 1), round(b.right / s + crop[0], 1), round(b.bottom / s + crop[1], 1)],
                area=a, conf=None if r.confidence is None else round(float(r.confidence), 3))
    v.unlink()
    o = dict(seeds=[[a, f, [float(x) for x in b]] for a, f, b in seeds], tracks=tr, res=res, wh=[w, h], direction=direction)
    json.dump(o, open(p, "w"))
    print(tag, "seeds", len(seeds), "tracked", len(tr), flush=True)
    return o


def has(mk, conf=CONF):
    return bool(mk and mk["area"] > 0 and (mk["conf"] or 0) >= conf)


def assign(stack, box, floor=None):
    """Truth stairs a seed box sits on: every stair whose box on any floor it hits. No best-of choice."""
    out = []
    for k, s in TRUTH[stack]["stairs"].items():
        bs = s["boxes"]
        if any(hit(box, b) for b in bs.values()):
            out.append(k)
    return out


def score(stack, tr, check=None, conf=CONF):
    """Strict per stair, per floor. A stair is seeded when at least one track's seed hits it; every such track must be right on every floor.
    `check(track_id, floor, mk) -> bool` optionally vetoes a mask. Returns rows and counts."""
    T = TRUTH[stack]; n = T["n"]
    seeds = {}
    for o, f, b in tr["seeds"]:
        seeds.setdefault(str(o), []).append((f, b))
    owner = {}
    for o, ss in seeds.items():
        for f, b in ss:
            for k in assign(stack, b, f):
                owner.setdefault(k, set()).add(o)
    rows = []
    for k, s in T["stairs"].items():
        if s["cls"] == "amb":
            continue
        ow = sorted(owner.get(k, ()))
        bx = {int(f): b for f, b in s["boxes"].items()}; ex = set(s.get("exclude", ()))
        for f in range(n):
            if f in ex:
                continue
            claims = []
            for o in ow:
                mk = tr["tracks"].get(o, {}).get(str(f))
                c = has(mk, conf) and (check is None or check(o, f, mk))
                claims.append((o, c, c and f in bx and hit(mk["box"], bx[f]), (mk or {}).get("conf")))
            if f in bx:
                ok = bool(ow) and all(c[2] for c in claims)
            else:
                ok = not any(c[1] for c in claims)
            rows.append(dict(stair=k, cls=s["cls"], floor=f, present=f in bx, seeded=bool(ow), ok=ok, ghosts=sum(c[1] for c in claims) if f not in bx else 0, claims=claims))
    false = [o for o, ss in seeds.items() if not any(assign(stack, b, f) for f, b in ss)]
    return rows, false


def summary(rows, classes=SCORED):
    r = [x for x in rows if x["cls"] in classes]
    pres = [x for x in r if x["present"]]; ab = [x for x in r if not x["present"]]
    st = {}
    for x in r:
        st.setdefault((x.get("stack"), x["stair"]), []).append(x)
    return dict(pairs=f"{sum(x['ok'] for x in r)}/{len(r)}", present=f"{sum(x['ok'] for x in pres)}/{len(pres)}", absent=f"{sum(x['ok'] for x in ab)}/{len(ab)}", ghosts=sum(x["ghosts"] for x in ab),
                stairs=f"{sum(all(y['ok'] for y in v) for v in st.values())}/{len(st)}", unseeded=sum(not v[0]["seeded"] for v in st.values()),
                base_pairs=f"{len(pres)}/{len(r)}", base_stairs=f"{sum(all(y['present'] for y in v) for v in st.values())}/{len(st)}")


# ---- Spark grounding, the prompt Round 1 used (lab/audit/heldout.py) ----
BOX = {"type": "object", "additionalProperties": False, "required": ["object_name", "x_min", "y_min", "x_max", "y_max"], "properties": {"object_name": {"type": "string"}, **{k: {"type": "integer"} for k in ("x_min", "y_min", "x_max", "y_max")}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["objects"], "properties": {"objects": {"type": "array", "items": BOX}}}
TAIL = ' For each object, answer in the format {"object_name": "<name>", "x_min": <int>, "y_min": <int>, "x_max": <int>, "y_max": <int>}. The coordinates should be in the 0-1000 range, x from the left edge and y from the top edge of the image. If you cannot find an object, omit it.'
DOC = 'You are an object grounding expert. Provide the bounding box coordinates of the objects in the image: "staircase". Ensure the objects accurately match the request and do not miss any objects.' + TAIL


def to_px(o, w, h):
    x1, x2 = sorted((o["x_min"], o["x_max"])); y1, y2 = sorted((o["y_min"], o["y_max"]))
    return [x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000]


def ground_floor(a):
    """Spark boxes for one floor of one stack, on the cropped frame, in stack px. Cached."""
    stack, f = a
    p = SPK / f"{stack}_f{f}.json"
    if p.exists():
        return json.load(open(p))
    x1, y1, x2, y2 = TRUTH[stack]["crop"]
    im = cv2.imread(str(ST / stack / f"f{f}.png"))[y1:y2, x1:x2]
    s = min(1.0, 2000 / max(im.shape[:2]))
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    q = R / "crops" / f"{stack}_f{f}.png"
    cv2.imwrite(str(q), im)
    r, info = m.spark(DOC, images=[str(q)], schema=SCHEMA)
    h, w = y2 - y1, x2 - x1
    o = dict(boxes=[[round(v, 1) for v in (b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1)] for b in (to_px(q, w, h) for q in r["objects"])], info=info)
    json.dump(o, open(p, "w"))
    print(stack, f, len(o["boxes"]), round(info["usd"], 4), flush=True)
    return o


def linked(a, b):
    area = lambda x: max(1, (x[2] - x[0]) * (x[3] - x[1])); cx = lambda x: ((x[0] + x[2]) / 2, (x[1] + x[3]) / 2)
    s, bg = (a, b) if area(a) < area(b) else (b, a); c = cx(s)
    return iou(a, b) >= .15 or (bg[0] <= c[0] <= bg[2] and bg[1] <= c[1] <= bg[3] and area(bg) < 5 * area(s))


def cluster(items):
    """Round 1 clustering across floors (lab/audit/heldout.py). items: (floor, box)."""
    par = list(range(len(items)))
    def find(i):
        while par[i] != i:
            i = par[i]
        return i
    for i in range(len(items)):
        for j in range(i):
            if linked(items[i][1], items[j][1]):
                par[find(i)] = find(j)
    g = {}
    for i in range(len(items)):
        g.setdefault(find(i), []).append(items[i])
    out = []
    for v in g.values():
        med = max(v, key=lambda x: sum(iou(x[1], y[1]) for y in v))
        out.append(dict(seed_floor=med[0], seed_box=med[1], members=v, floors=sorted({f for f, _ in v})))
    return sorted(out, key=lambda c: -len(c["floors"]))


if __name__ == "__main__":
    assert hit([0, 0, 10, 10], [0, 0, 10, 10]) and not hit([100, 100, 110, 110], [0, 0, 10, 10])
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9
    print("core ok", STACKS, "spent", m.spent("tightseed"))
