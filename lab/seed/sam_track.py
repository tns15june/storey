"""SAM video tracking of Spark seeds through a floor stack, with padding and refinement variants. Cached per run.
Usage: sam_track.py matrix"""
import sys; sys.path.insert(0, "lab/seed"); sys.path.insert(0, "lab")
from common import *
from concurrent.futures import ThreadPoolExecutor
import metalab as m

REPEAT = 4
FRAMES = {}


def frames(bid):
    if bid not in FRAMES:
        FRAMES[bid] = [cv2.imread(str(OUT / bid / f"f{i}.png")) for i in range(len(CAT[bid]["floors"]))]
        for f in FRAMES[bid]:
            assert f.shape == FRAMES[bid][0].shape
    return FRAMES[bid]


def video(bid):
    p = OUT / bid / "stack.mp4"
    fr = frames(bid)
    h, w = fr[0].shape[:2]
    wh = (w - w % 2, h - h % 2)
    if not p.exists():
        assert stack_mp4(fr, p, REPEAT) == wh
    return p, wh


def spark_boxes(bid, floor, var="plan", cls=None):
    w, h = video(bid)[1]
    obs = json.load(open(OUT / "spark" / f"{bid}_f{floor}_{var}.json"))["objects"]
    return [to_px(o, w, h) for o in obs if cls is None or o["object_name"].lower() == cls]


def pad(b, p, wh):
    w, h = wh
    dx, dy = (b[2] - b[0]) * p, (b[3] - b[1]) * p
    return [max(0, round(b[0] - dx)), max(0, round(b[1] - dy)), min(w - 1, round(b[2] + dx)), min(h - 1, round(b[3] + dy))]


def token(b, wh):
    return f"<|box;x1={b[0]};y1={b[1]};x2={b[2]};y2={b[3]};w={wh[0]};h={wh[1]}|>"


def refine(bid, floor, boxes):
    """Spark box to SAM image call, one request per 16 boxes, returns the mask's own bounds or the input box if SAM found nothing."""
    wh = video(bid)[1]
    im = frames(bid)[floor][:wh[1], :wh[0]]
    out = list(boxes)
    for s in range(0, len(boxes), 16):
        part = boxes[s:s + 16]
        prompt = "<0f>" + ",".join(f"{s + i + 1}{token(b, wh)}" for i, b in enumerate(part))
        for r in m.sam_image(prompt, im):
            b = r.bounds
            out[int(r.object_id) - 1] = [b.left, b.top, b.right, b.bottom]
    return out


def run(tag, bid, seeds):
    """seeds: list of (object id, floor, box). Returns {obj: {floor: {box, area, conf, n}}} from the last frame of each floor block."""
    path = OUT / "sam" / f"{tag}.json"
    if path.exists():
        return json.load(open(path))
    mp4, wh = video(bid)
    n = len(frames(bid)) * REPEAT
    tracks, masks = {}, {}
    for s in range(0, len(seeds), 16):
        part = seeds[s:s + 16]
        byf = {}
        for oid, f, b in part:
            byf.setdefault(f, []).append(f"{oid}{token(b, wh)}")
        prompt = "".join(f"<{f * REPEAT}f>" + ",".join(v) for f, v in sorted(byf.items()))
        for r in m.sam_video(prompt, mp4, frames=n):
            fl, sub = r.frame.frame_index // REPEAT, r.frame.frame_index % REPEAT
            t = tracks.setdefault(str(r.object_id), {}).setdefault(str(fl), {"n": 0})
            t["n"] += 1
            if sub == REPEAT - 1:
                b = r.bounds
                rast = m.raster(r)
                t.update(box=[b.left, b.top, b.right, b.bottom], area=int(rast.sum()), conf=None if r.confidence is None else round(r.confidence, 3))
                if tag.startswith("union"):
                    masks[f"{r.object_id}_{fl}"] = rast
    if masks:
        np.savez_compressed(OUT / "sam" / f"{tag}.npz", **masks)
    out = {"bid": bid, "seeds": [[o, f, list(map(int, b))] for o, f, b in seeds], "tracks": tracks}
    json.dump(out, open(path, "w"))
    print(tag, "objects", len(seeds), "tracked", len(tracks), flush=True)
    return out


def matrix():
    jobs = []
    for bid in ("bme_f", "schenley", "habs_stelizabeths_atkins", "habs_stelizabeths_center"):
        nf = len(CAT[bid]["floors"])
        for f in range(nf):
            bx = spark_boxes(bid, f)
            jobs.append((f"{bid}__f{f}__p0", bid, [(i + 1, f, b) for i, b in enumerate(bx)]))
        # padding and refinement are compared on the two most populated seed floors
        fl = sorted(range(nf), key=lambda f: -len([t for t in TRUTH[bid] if f in t["floors"] and t["cls"] == "main"]))[:1] + [nf - 1]
        wh = video(bid)[1]
        for f in dict.fromkeys(fl):
            bx = spark_boxes(bid, f)
            for p in (0.1, 0.25, 0.5):
                jobs.append((f"{bid}__f{f}__p{p}", bid, [(i + 1, f, pad(b, p, wh)) for i, b in enumerate(bx)]))
            tag = f"{bid}__f{f}__refine"
            if not (OUT / "sam" / f"{tag}.json").exists():
                rb = refine(bid, f, [pad(b, 0.1, wh) for b in bx])
                jobs.append((tag, bid, [(i + 1, f, b) for i, b in enumerate(rb)]))
    with ThreadPoolExecutor(5) as ex:
        list(ex.map(lambda j: run(*j), jobs))


if __name__ == "__main__":
    (OUT / "sam").mkdir(exist_ok=True)
    if sys.argv[1] == "matrix":
        matrix()
