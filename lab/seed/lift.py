"""Elevators and toilets as tracked objects, plus a planted shift test: can tracks measure a known misregistration?"""
import sys; sys.path.insert(0, "lab/seed"); sys.path.insert(0, "lab")
from common import *
from union import cluster, CONF
from sam_track import run, video, frames, REPEAT, token
import sam_track
import metalab as m
import subprocess
from concurrent.futures import ThreadPoolExecutor

BIDS = ["habs_stelizabeths_center", "habs_stelizabeths_atkins", "nibs_clinic", "bme_f"]


def lifts(bid):
    cl = cluster(bid, "lift")
    names = {}
    for c in cl:
        pass
    r = run(f"union_lift__{bid}", bid, [(i + 1, c["seed_floor"], c["seed_box"]) for i, c in enumerate(cl)])
    json.dump({"clusters": cl, "gate": {}, "summary": {"raw": {"rows": {}}}}, open(OUT / "sam" / f"result_lift__{bid}.json", "w"), indent=1, default=int)
    print(bid, "lift clusters", len(cl), [(c["floors"]) for c in cl][:12])


def center(tr, f):
    m_ = tr.get(str(f))
    if not m_ or "box" not in m_ or (m_.get("conf") or 0) < CONF:
        return None
    b = m_["box"]
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def shift_test(bid, floor, dx, dy, tag_seeds):
    """Move one floor by a planted offset, track, and read the offset back from the tracks."""
    tag = f"shift_{bid}_f{floor}_{dx}_{dy}"
    path = OUT / "sam" / f"{tag}.json"
    mp4 = OUT / bid / f"{tag}.mp4"
    fr = [f.copy() for f in frames(bid)]
    h, w = fr[0].shape[:2]
    fr[floor] = cv2.warpAffine(fr[floor], np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderValue=(255, 255, 255))
    if not path.exists():
        stack_mp4(fr, mp4, REPEAT)
        wh = (w - w % 2, h - h % 2)
        recs = {}
        for s in range(0, len(tag_seeds), 16):
            part = tag_seeds[s:s + 16]
            byf = {}
            for oid, f, b in part:
                byf.setdefault(f, []).append(f"{oid}{token(b, wh)}")
            prompt = "".join(f"<{f * REPEAT}f>" + ",".join(v) for f, v in sorted(byf.items()))
            for r in m.sam_video(prompt, mp4, frames=len(fr) * REPEAT):
                fl, sub = r.frame.frame_index // REPEAT, r.frame.frame_index % REPEAT
                if sub == REPEAT - 1:
                    b = r.bounds
                    recs.setdefault(str(r.object_id), {})[str(fl)] = {"box": [b.left, b.top, b.right, b.bottom], "conf": r.confidence}
        json.dump(recs, open(path, "w"))
        mp4.unlink()
    return json.load(open(path))


def shift_all():
    out = []
    for bid, floor, shifts in (("bme_f", 2, [(40, 25), (90, 60), (180, 120)]), ("habs_stelizabeths_center", 2, [(40, 25), (90, 60), (180, 120)]), ("schenley", 1, [(40, 25), (90, 60), (180, 120)])):
        res = json.load(open(OUT / "sam" / f"result_doc__{bid}.json"))
        rows = res["summary"]["raw"]["rows"]
        keep = sorted({r["cluster"] for r in rows.values() if r["cls"] == "main" and r.get("all_ok")})
        seeds = [(i + 1, res["clusters"][i]["seed_floor"], res["clusters"][i]["seed_box"]) for i in keep if res["clusters"][i]["seed_floor"] != floor]
        base = json.load(open(OUT / "sam" / f"union_doc__{bid}.json"))["tracks"]
        for dx, dy in shifts:
            tr = shift_test(bid, floor, dx, dy, seeds)
            est = []
            kept = 0
            for oid, f, b in seeds:
                a, c = center(tr.get(str(oid), {}), floor), center(base.get(str(oid), {}), floor)
                if a and c:
                    kept += 1
                    est.append((a[0] - c[0], a[1] - c[1]))
            med = np.median(est, axis=0).round(1).tolist() if est else None
            row = {"bid": bid, "floor": floor, "planted": [dx, dy], "objects": len(seeds), "tracked_at_conf": kept, "median_read_back": med,
                   "error_px": None if med is None else round(float(np.hypot(med[0] - dx, med[1] - dy)), 1)}
            print(row, flush=True)
            out.append(row)
    json.dump(out, open(OUT / "shift_test.json", "w"), indent=1)


if __name__ == "__main__":
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lifts, BIDS))
    shift_all()
