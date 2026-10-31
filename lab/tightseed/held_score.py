"""Score held-out runs under the frozen rule. Wrong passes print first. No API."""
import sys, glob
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import dev1, seeds as sd, multi as mu, tread

STAIRS = HELD
TAGS = {"hand_960": None, "hand_2000": None, "loose_960": None, "snap_960": None, "snap_2000": None, "sparkcrop_960": None, "vector_960": None, "d_960": None}


def load(tag):
    p = SAMD / f"{tag}.json"
    return json.load(open(p)) if p.exists() else None


def tread_checker(st, gr, memo):
    def chk(o, f, mk):
        k = (st, f, tuple(mk["box"]))
        if k not in memo:
            memo[k] = mu.tread_ok(gr[f], mk["box"])[0]
        return memo[k]
    return chk


def spark_checker(st, tag):
    def chk(o, f, mk):
        p = SPK / f"{st}_chk{tag}_{o}_{f}.json"
        return json.load(open(p))["stair"] if p.exists() else True
    return chk


def tag_rows(st, rows):
    for r in rows:
        r["stack"] = st
    return rows


def evote(st):
    """Rebuild method e's vote from its cached track."""
    tr = load(f"{st}__e_960")
    if tr is None:
        return None
    cl = json.load(open(SPK / f"{st}_clusters.json"))
    gr = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames(st)]
    # same call as dev: method_e reads the cache and never reaches the API when the tag exists
    return mu.method_e(f"{st}__e_960", st, cl, gr)


def spark_member_track(st):
    cl = json.load(open(SPK / f"{st}_clusters.json"))
    tracks, seeds = {}, []
    for i, c in enumerate(cl):
        seeds.append([i + 1, c["seed_floor"], c["seed_box"]])
        tracks[str(i + 1)] = {str(f): dict(box=b, area=1, conf=1.0) for f, b in c["members"]}
    return dict(seeds=seeds, tracks=tracks)


def seed_iou_rows(st, tr):
    """Each stair: every track seeded on it, IoU on the seed floor against the hand box. Unseeded stairs give a 0."""
    out = []
    T = TRUTH[st]["stairs"]
    seeded = {}
    for o, f, b in tr["seeds"]:
        for k in assign(st, b):
            seeded.setdefault(k, []).append((f, b))
    for k, s in T.items():
        if s["cls"] not in ("main", "short", "ext"):
            continue
        if k not in seeded:
            out.append((st, k, s["cls"], None, 0.0)); continue
        for f, b in seeded[k]:
            t = s["boxes"].get(str(f))
            out.append((st, k, s["cls"], f, round(iou(b, t), 3) if t else 0.0))
    return out


def anyconf_ghosts(rows, tr, st):
    """Masks of any confidence on truth-absent floors, for tracks seeded on the stair."""
    n = 0
    owners = {}
    for o, f, b in tr["seeds"]:
        for k in assign(st, b):
            owners.setdefault(k, set()).add(str(o))
    for r in rows:
        if not r["present"]:
            for o in owners.get(r["stair"], ()):
                mk = tr["tracks"].get(o, {}).get(str(r["floor"]))
                n += bool(mk and mk["area"] > 0)
    return n


if __name__ == "__main__":
    methods = ["hand_960", "hand_2000", "loose_960", "snap_960", "snap_2000", "sparkcrop_960", "vector_960", "d_960"]
    table = {}; iou_rows = {}; all_rows = {}
    memo = {}
    GR = {st: [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames(st)] for st in HELD}
    for me in methods:
        for st in HELD:
            tr = load(f"{st}__{me}")
            if tr is None:
                continue
            for lab, chk in (("raw", None), ("tread", tread_checker(st, GR[st], memo)), ("spark", spark_checker(st, me))):
                if lab == "spark" and not glob.glob(str(SPK / f"{st}_chk{me}_*.json")):
                    continue
                rows, false = score(st, tr, chk)
                all_rows.setdefault((me, lab), []).extend(tag_rows(st, rows))
                table.setdefault((me, lab), {})[st] = dict(summary(rows), false=len(false))
            iou_rows.setdefault(me, []).extend(seed_iou_rows(st, tr))
    for st in HELD:
        e = evote(st)
        if e:
            rows, false = score(st, e)
            all_rows.setdefault(("e_960", "vote"), []).extend(tag_rows(st, rows))
            table.setdefault(("e_960", "vote"), {})[st] = dict(summary(rows), false=len(false))
        sm = spark_member_track(st)
        rows, false = score(st, sm)
        all_rows.setdefault(("spark_members", "nosam"), []).extend(tag_rows(st, rows))
        table.setdefault(("spark_members", "nosam"), {})[st] = dict(summary(rows), false=len(false))
        seeds = json.load(open(SPK / f"{st}_seeds.json"))
        for seedname in ("loose", "snap", "sparkcrop"):
            sb = [(o, f, b, fb) for o, f, b, fb in seeds[seedname]]
            for lab, fn in (("nosam_tread", mu.no_sam), ("nosam_ink", mu.no_sam_ink)):
                rows, false = score(st, fn(st, sb, GR[st]))
                all_rows.setdefault((seedname, lab), []).extend(tag_rows(st, rows))
                table.setdefault((seedname, lab), {})[st] = dict(summary(rows), false=len(false))
    print("=== WRONG PASSES FIRST: scored stairs (main, short), truth-absent floors, claims that survive ===")
    for k, rows in all_rows.items():
        g = [r for r in rows if r["cls"] in SCORED and not r["present"] and not r["ok"]]
        print(k, "wrong passes", len(g), [(r["stack"], r["stair"], r["floor"], [c[3] for c in r["claims"] if c[1]]) for r in g])
    print("\n=== POOLED strict per stair per floor, main+short ===")
    for k, rows in all_rows.items():
        s = summary(rows)
        print(f"{k[0]:16s}{k[1]:12s}", s, "false tracks", sum(v["false"] for v in table[k].values()))
    print("\n=== by stack ===")
    for k, v in table.items():
        print(k, {st: (x["pairs"], "abs " + x["absent"], "pres " + x["present"], "unseeded", x["unseeded"], "false", x["false"]) for st, x in v.items()})
    print("\n=== seed IoU, stairs of all classes, unseeded = 0 ===")
    for me, v in iou_rows.items():
        a = np.array([x[4] for x in v if x[2] in ("main", "short")])
        print(me, "n", len(a), "median", round(float(np.median(a)), 3), ">=.5", int((a >= .5).sum()), ">=.3", int((a >= .3).sum()), "zero", int((a == 0).sum()))
        for st in HELD:
            b = np.array([x[4] for x in v if x[0] == st and x[2] in ("main", "short")])
            if len(b):
                print("    ", st, "n", len(b), "median", round(float(np.median(b)), 3), ">=.5", int((b >= .5).sum()))
    json.dump({f"{a}|{b}": v for (a, b), v in table.items()}, open(R / "table.json", "w"), indent=1)
    print("spent", round(m.spent("tightseed"), 4))
