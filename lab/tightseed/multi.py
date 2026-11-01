"""Methods d and e, the Spark crop check, and the no-SAM registered-box checks."""
import sys
sys.path.insert(0, "lab/tightseed")
from core import *
import seeds as sd, tread, reg


def member_seeds(cl, gr, grow=.15):
    """Per cluster, a tight box for each member: its own snap on its own floor, or the member box when snap finds nothing."""
    out = []
    for i, c in enumerate(cl):
        ms = []
        for f, b in c["members"]:
            s, _ = tread.snap(gr[f], b, grow=grow)
            ms.append((f, s or b))
        out.append(ms)
    return out


def method_d(tag, st, cl, gr):
    """One object id, pinned on every floor the cluster has a member. A cluster with two boxes on one floor keeps the first."""
    seeds = []
    for i, ms in enumerate(member_seeds(cl, gr)):
        seen = set()
        for f, b in ms:
            if f not in seen:
                seeds.append((i + 1, f, b)); seen.add(f)
    return track(tag, st, seeds)


def method_e(tag, st, cl, gr):
    """One object per member, so each floor's seed tracks alone. Votes back to one track per cluster: a floor is claimed when at least half the cluster's tracks have a mask there."""
    seeds, groups = [], {}
    k = 0
    for i, ms in enumerate(member_seeds(cl, gr)):
        for f, b in ms:
            k += 1; seeds.append((k, f, b)); groups.setdefault(i + 1, []).append(str(k))
    tr = track(tag, st, seeds)
    votes = {}
    n = TRUTH[st]["n"]
    for g, ids in groups.items():
        votes[str(g)] = {}
        for f in range(n):
            mk = [tr["tracks"].get(o, {}).get(str(f)) for o in ids]
            on = [x for x in mk if has(x)]
            if 2 * len(on) >= len(ids):
                a = np.array([x["box"] for x in on])
                votes[str(g)][str(f)] = dict(box=np.median(a, 0).tolist(), area=1, conf=1.0)
    return dict(seeds=[[g, tr["seeds"][int(ids[0]) - 1][1], tr["seeds"][int(ids[0]) - 1][2]] for g, ids in groups.items()], tracks=votes, raw=tr)


CHECK_P = ("This is a crop of an architectural floor plan. Is a staircase drawn inside the red rectangle, meaning a run of parallel tread lines showing steps? "
           "A plain room, corridor, text, hatching, fixtures, a shaft or blank paper is not a staircase. Answer with JSON.")
CHECK_S = {"type": "object", "additionalProperties": False, "required": ["stair"], "properties": {"stair": {"type": "boolean"}}}


def spark_yes(a):
    """a = (stack, key, frame, box). One Spark verdict on a crop with the box drawn in. Cached."""
    st, key, fr, b = a
    p = SPK / f"{st}_chk{key}.json"
    if p.exists():
        return json.load(open(p))["stair"]
    x1, y1, x2, y2 = sd.crop_window(b, fr.shape, k=1.6, minside=200)
    c = fr[y1:y2, x1:x2].copy()
    z = min(4.0, 800 / max(c.shape[:2]))
    c = cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA)
    cv2.rectangle(c, (int((b[0] - x1) * z), int((b[1] - y1) * z)), (int((b[2] - x1) * z), int((b[3] - y1) * z)), (0, 0, 255), 2)
    q = R / "crops" / f"{st}_chk{key}.png"
    cv2.imwrite(str(q), c)
    r, info = m.spark(CHECK_P, images=[str(q)], schema=CHECK_S)
    json.dump(dict(stair=r["stair"], info=info), open(p, "w"))
    return r["stair"]


def tread_ok(g, box):
    r = tread.score_box(g, box)
    return r["steps"] >= RULE["steps"] and r["q90"] >= RULE["q90"], r


RULE = dict(steps=5, q90=.4)


def no_sam(st, cl_seeds, gr):
    """No SAM at all: the seed box, unmoved, on every floor of the registered stack. Returns a track dict for score()."""
    n = TRUTH[st]["n"]; tracks = {}
    for o, f0, b, _ in cl_seeds:
        tracks[str(o)] = {}
        for f in range(n):
            ok, r = tread_ok(gr[f], b)
            if ok:
                tracks[str(o)][str(f)] = dict(box=list(b), area=1, conf=1.0)
    return dict(seeds=[[o, f, list(b)] for o, f, b, _ in cl_seeds], tracks=tracks)


def inkmap(g):
    return cv2.GaussianBlur(g.astype(np.float32), (0, 0), 8) - g.astype(np.float32) > 12


def no_sam_ink(st, cl_seeds, gr, ratio=.45):
    """Round 1 ink test: ink density in the seed box against its seed floor."""
    n = TRUTH[st]["n"]; tracks = {}
    ink = [inkmap(g) for g in gr]
    for o, f0, b, _ in cl_seeds:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        d = [float(ink[f][y1:y2, x1:x2].mean()) if y2 > y1 and x2 > x1 else 0 for f in range(n)]
        tracks[str(o)] = {str(f): dict(box=list(b), area=1, conf=1.0) for f in range(n) if d[f] >= ratio * max(d[f0], 1e-6)}
    return dict(seeds=[[o, f, list(b)] for o, f, b, _ in cl_seeds], tracks=tracks)
