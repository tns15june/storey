"""Item 5: Spark reads a 1 fps floor-stack MP4 instead of images. Scored with the lab/audit/verify.py one-to-one match. Usage: item5.py run <runs> <long side> | score"""
import json
import subprocess
import sys

sys.path.insert(0, "lab/samtricks")
import api as a  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import reg  # noqa: E402
import rules as R  # noqa: E402

A = reg.ROOT / "data/lab/audit"
T = json.load(open(reg.ROOT / "lab/audit/truth.json"))
NF = {"habs_va_bldg1": 5, "bme_k": 4, "wfu_huffman": 4}
ORDER = {"habs_va_bldg1": "ground, first, second, third, fourth", "bme_k": "lower ground, ground, first, second", "wfu_huffman": "ground, first, second, third"}
OUT = reg.OUT / "item5"
OUT.mkdir(exist_ok=True)
STR, INT = {"type": "string"}, {"type": "integer"}


def obj(**k):
    return {"type": "object", "additionalProperties": False, "required": list(k), "properties": k}


def arr(i):
    return {"type": "array", "items": i}


SCHEMA = obj(frames_seen=INT, stairs=arr(obj(name=STR, floors=arr(obj(floor=STR, present={"type": "boolean"}, x=INT, y=INT)))))
# The image prompt from lab/audit/heldout.py with its first sentence changed for video, and a frame count added.
PROMPT = """This video shows floor plans of one building, one frame per floor, each held for one second, in this order: {order}. Staircases in the same place on consecutive floors are the same staircase.
List every distinct staircase in the building. For each give a short name, and for every floor in the list say whether it is drawn on that floor (present) and its centre x, y in the 0-1000 range of that floor's frame (0, 0 if absent). A stair drawn struck through or filled in counts as absent. Compare the floors carefully, stair by stair.
Also give frames_seen, the number of distinct floor frames you saw in the video."""


def frames(b):
    return [cv2.imread(str(A / b / f"f{i}.png")) for i in range(NF[b])]


def mp4(b, long_side):
    p = OUT / f"{b}_{long_side}.mp4"
    if p.exists():
        return p
    fr = frames(b)
    s = long_side / max(fr[0].shape[:2])
    w, h = int(fr[0].shape[1] * s) & ~1, int(fr[0].shape[0] * s) & ~1
    e = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "1", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(p)], stdin=subprocess.PIPE)
    for f in fr:
        e.stdin.write(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA).tobytes())
    e.stdin.close()
    e.wait()
    return p


def run(b, run_id, long_side):
    p = OUT / f"{b}_{long_side}_run{run_id}.json"
    if p.exists():
        return json.load(open(p))
    r, info = a.chat([a.text(PROMPT.format(order=ORDER[b])), a.video(mp4(b, long_side), fps=1)], schema=SCHEMA, effort="medium", max_tokens=20000, kind="item5")
    o = dict(out=r, info=info, mb=round(mp4(b, long_side).stat().st_size / 1e6, 2))
    p.write_text(json.dumps(o, indent=1))
    return o


def wh(b):
    im = cv2.imread(str(A / b / "f0.png"))
    h, w = im.shape[:2]
    return w - w % 2, h - h % 2


def score(b, fl):
    """verify.py one-to-one match. Returns counts for main-class stairs."""
    w, h = wh(b)
    t = {k: v for k, v in T[b].items() if v["cls"] == "main"}
    nf = NF[b]
    pairs = []
    for k, s in t.items():
        bx = {int(f): v for f, v in s["boxes"].items()}
        for i, st in enumerate(fl):
            if len(st["floors"]) != nf:
                continue
            d = [np.hypot(st["floors"][f]["x"] * w / 1000 - (bx[f][0] + bx[f][2]) / 2, st["floors"][f]["y"] * h / 1000 - (bx[f][1] + bx[f][3]) / 2) for f in bx if st["floors"][f]["present"]]
            if d:
                pairs.append((float(np.median(d)), k, i))
    used, got = set(), {}
    for d, k, i in sorted(pairs):
        if k in got or i in used or d > 120:
            continue
        got[k] = i
        used.add(i)
    c = dict(pp=0, pn=0, ap=0, an=0, stacks=0, n=0, wrong_present=0, unmatched=0, listed=len(fl), unused=len(fl) - len(used))
    bad = []
    for k, s in t.items():
        bx = {int(f): v for f, v in s["boxes"].items()}
        c["n"] += 1
        if k not in got:
            bad.append((k, "unmatched"))
            c["unmatched"] += 1
            c["pn"] += len(bx)
            c["an"] += nf - len(bx)
            continue
        st = fl[got[k]]
        this = True
        for f in range(nf):
            p = bool(st["floors"][f]["present"])
            if f in bx:
                c["pn"] += 1
                c["pp"] += p
            else:
                c["an"] += 1
                c["ap"] += not p
                c["wrong_present"] += p
            if p != (f in bx):
                this = False
                bad.append((k, f, p))
        c["stacks"] += this
    return c, bad


def add(x, y):
    for k in y:
        x[k] = x.get(k, 0) + y[k]
    return x


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "run":
        runs, ls = int(sys.argv[2]), int(sys.argv[3])
        rows = a.pmap(lambda j: (j, run(j[0], j[1], ls)), [(b, r) for b in R.VIDEO_BUILDINGS for r in range(runs)], 2)
        for j, o in rows:
            print(j, o["info"], "frames_seen", o["out"]["frames_seen"], "stairs", len(o["out"]["stairs"]), "mb", o["mb"])
    print("no-model always-present: pairs right, stacks right")
    tot_b = {}
    for b in R.VIDEO_BUILDINGS:
        t = {k: v for k, v in T[b].items() if v["cls"] == "main"}
        pres = sum(len(s["boxes"]) for s in t.values())
        tot = NF[b] * len(t)
        print(" ", b, f"{pres}/{tot}", sum(len(s["boxes"]) == NF[b] for s in t.values()), "/", len(t))
    print("image runs, cached from Round 1")
    ti = {}
    usd = []
    for b in R.VIDEO_BUILDINGS:
        for r in (0, 1, 2):
            o = json.load(open(A / "spark" / f"{b}_floors{r}.json"))
            c, bad = score(b, o["out"]["stairs"])
            add(ti, c)
            usd.append(o["info"]["usd"])
            print(" ", b, r, c, bad, round(o["info"]["usd"], 4), o["info"]["input"], o["info"]["output"])
    print(" total", ti, "mean usd", round(float(np.mean(usd)), 4))
    for f in sorted(OUT.glob("*_run*.json")):
        b = f.name.split("_run")[0].rsplit("_", 1)[0]
        o = json.load(open(f))
        c, bad = score(b, o["out"]["stairs"])
        print(f.name, c, bad, "frames_seen", o["out"]["frames_seen"], "usd", round(o["info"]["usd"], 4), "in", o["info"]["input"], "out", o["info"]["output"])
    print("spent", round(a.m.spent("samtricks"), 3))
