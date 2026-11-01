"""Numbers for the report, from cached evidence. No API. Added after held-out, so it changes no decision: it only reads the frozen rule's output more ways.
Usage: report.py dev|heldout"""
import json
import sys
from collections import Counter

sys.path.insert(0, "lab/presence")
import numpy as np

import cells as C
import planted
import pres
import rule
import score as S
from pres import OUT, T, apply_pt, box_dist, ctr, fam

split = sys.argv[1]
B = [b for b, t in T.items() if t["split"] == split]
ARMS = rule.ARMS


def cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound, by bisection on the binomial tail."""
    from math import comb
    if n == 0:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        tail = sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
        lo, hi = (p, hi) if tail > 1 - conf else (lo, p)
    return round(hi, 3)


def where(b, v, f):
    """Registered place of a truth object on floor f, from the floors where it is drawn."""
    A = pres.alignment(b)
    if not A[str(f)]["aligned"]:
        return None
    p = S.ref_pos(b, v)
    return None if p is None else apply_pt(np.linalg.inv(np.array(A[str(f)]["H"])), p)


print("ABSENT CELLS. Truth says the object is not drawn on this floor. Worst state of any row at its registered place, within", S.D, "px.")
tab = {}
detail = []
for b in B:
    E = json.load(open(OUT / f"cells/{b}.json"))
    dec = {a: rule.apply_rule(E, a) for a in ARMS}
    for o, v in T[b]["objects"].items():
        F = fam(v["kind"])
        us = {int(x) for x in v.get("unsure", [])}
        for f in range(pres.nfloors(b)):
            if str(f) in v["boxes"] or f in us:
                continue
            q = where(b, v, f)
            for a in ARMS:
                k = (F, v["cls"], a)
                t = tab.setdefault(k, Counter())
                if q is None:
                    t["unaligned or no place"] += 1
                    continue
                here = [c for c in dec[a] if c["fam"] == F and c["f"] == f and c["state"] and c["chk"] and box_dist(c["chk"], (q[0], q[1], q[0], q[1])) <= S.D]
                st = [c["state"] for c in here]
                w = "Found" if "Found" in st else ("Check" if "Check" in st else ("Missing" if "Missing" in st else "no row there"))
                t[w] += 1
                if w == "Found":
                    detail.append((b, o, f, a, [round(x) for x in q], [c["v"] for c in here if c["state"] == "Found"]))
for k, t in sorted(tab.items()):
    print(" ", k, dict(t))
print(" Found on an absent cell:", detail)
for a in ARMS:
    t = Counter()
    for k, c in tab.items():
        if k[2] == a and k[0] == "stair":
            t.update(c)
    tested = t["Found"] + t["Check"] + t["Missing"]
    print(f" stairs arm {a}: tested {tested} Found {t['Found']} upper95 {cp_upper(t['Found'], tested)}; no row there {t['no row there']}")

print("\nPRESENT CELLS by class and kind, arm by arm. Best state of any row near the true box.")
for a in ARMS:
    t = {}
    for b in B:
        E = json.load(open(OUT / f"cells/{b}.json"))
        dec = rule.apply_rule(E, a)
        for o, v in T[b]["objects"].items():
            F = fam(v["kind"])
            us = {int(x) for x in v.get("unsure", [])}
            for f_, bx in v["boxes"].items():
                f = int(f_)
                if f in us:
                    continue
                here = [c["state"] for c in dec if c["fam"] == F and c["f"] == f and c["state"] and c["loc"] and box_dist(c["loc"], bx) <= S.D]
                w = "Found" if "Found" in here else ("Check" if "Check" in here else ("Missing" if "Missing" in here else "not listed"))
                t.setdefault((v["kind"], v["cls"]), Counter())[w] += 1
    print(" arm", a, {k: dict(v) for k, v in sorted(t.items())})
c0 = {}
for b in B:
    d = S.single_run(b, b, 0)
    for o, v in T[b]["objects"].items():
        if fam(v["kind"]) != "stair" or v["cls"] != "main":
            continue
        for f_, bx in v["boxes"].items():
            here = [c["state"] for c in d if c["f"] == int(f_) and box_dist(c["loc"], bx) <= S.D]
            w = "Found" if "Found" in here else ("Missing" if "Missing" in here else "not listed")
            c0[w] = c0.get(w, 0) + 1
print(" run0 main stairs", c0)

print("\nPLANTED cells: evidence, plus any row whose Spark point or registered place sits on the erased box.")
keys = planted.DEV_KEYS if split == "dev" else planted.HELD_KEYS
catch = Counter()
rows_ = []
coll = {a: Counter() for a in ARMS}
for key, (b, cases) in keys.items():
    p = OUT / f"cells/{key}.json"
    if not p.exists():
        continue
    E = json.load(open(p))
    for kind, o, f in cases:
        bx = T[b]["objects"][o]["boxes"][str(f)]
        hit = []
        for r in E["rows"]:
            if r["fam"] != "stair":
                continue
            c = r["cells"][str(f)]
            near_ = [box_dist(x, bx) for x in (c["loc"], c["chk"]) if x]
            if near_ and min(near_) <= S.D:
                hit.append(dict(v=c["v"], crop=c.get("crop"), ink=r["ink30"].get(str(f)), blank=c.get("blank"), aligned=c["aligned"], est=r["established"], seen=str(c.get("seen"))[:60], states={a: rule.decide(dict(r, ink=r["ink30"].get(str(f))), c, a)[0] for a in ARMS}))
        st = {a: ("Found" if any(h["states"][a] == "Found" for h in hit) else ("Check" if any(h["states"][a] == "Check" for h in hit) else ("Missing" if any(h["states"][a] == "Missing" for h in hit) else "no row"))) for a in ARMS}
        print(f" {key:40s} {kind:9s} {o:10s} f{f} {st} rows {[(h['v'], h['crop'], h['ink'], h['blank']) for h in hit]}")
        rows_.append(dict(key=key, b=b, kind=kind, obj=o, f=f, st=st, hit=hit))
        for h in hit:
            if h["v"] >= 2 and h["aligned"] and h["est"]:
                catch["cells v>=2"] += 1
                catch["crop no/unclear"] += h["crop"] in ("no", "unclear")
                catch["ink under 0.40"] += h["ink"] is not None and h["ink"] < rule.INK_LOW
                catch["crop or ink"] += h["crop"] in ("no", "unclear") or (h["ink"] is not None and h["ink"] < rule.INK_LOW)
                catch["crop yes"] += h["crop"] == "yes"
    for a in ARMS:
        s = S.score(b, [c for c in rule.apply_rule(E, a) if c["fam"] == "stair"], "stair")
        erased = [(c[1], c[2]) for c in cases]
        coll[a]["wrong elsewhere"] += len(s["wrong"])
        coll[a]["phantom"] += len(s["phantom"])
        coll[a]["false, not the erased cell"] += sum(1 for x in s["false"] if (x["obj"], x["f"]) not in erased)
        coll[a]["check"] += s["check"]
print(" evidence on planted cells that 2 or 3 runs called present:", dict(catch))
print(" collateral on planted stacks, whole stack:", {a: dict(v) for a, v in coll.items()})
for a in ARMS + ("run0",):
    if a != "run0":
        k = sum(r["st"][a] == "Found" for r in rows_)
        print(f" arm {a}: wrong passes {k} of {len(rows_)}, upper95 {cp_upper(k, len(rows_))}")

print("\nCLEAN stacks rescored at the registered row place instead of Spark's point. Post hoc: score.py is unchanged, this passes chk as the place.")
for F in ("stair", "shaft"):
    for a in ARMS:
        t = Counter()
        lst = []
        for b in B:
            E = json.load(open(OUT / f"cells/{b}.json"))
            d = [dict(c, loc=c["chk"]) for c in rule.apply_rule(E, a) if c["fam"] == F]
            s = S.score(b, d, F)
            t["wrong"] += len(s["wrong"]); t["phantom"] += len(s["phantom"]); t["false"] += len(s["false"]); t["check"] += s["check"]; t["found ok"] += s["ok"]
            lst += [(b, x["obj"], x["f"], x["loc"]) for x in s["wrong"] + s["phantom"] + s["false"]]
        print(f" {F} arm {a}", dict(t), lst)
print(" Found at tighter tolerance, arm C stairs, by marker and by registered place: phantoms if D were 20 or 0")
sv = S.D
for d_ in (20, 0):
    S.D = d_
    n1 = n2 = 0
    for b in B:
        E = json.load(open(OUT / f"cells/{b}.json"))
        dec = [c for c in rule.apply_rule(E, "C") if c["fam"] == "stair"]
        s1 = S.score(b, dec, "stair"); s2 = S.score(b, [dict(c, loc=c["chk"]) for c in dec], "stair")
        n1 += len(s1["phantom"]) + len(s1["wrong"]); n2 += len(s2["phantom"]) + len(s2["wrong"])
    print(f"  D {d_}: by marker {n1}, by registered place {n2} of Found 75")
S.D = sv

print("\nCHECK reasons, clean, arm C, by what is really at the place.")
t = Counter()
for b in B:
    E = json.load(open(OUT / f"cells/{b}.json"))
    for c in rule.apply_rule(E, "C"):
        if c["state"] == "Check":
            pr, un = S.near(b, c["f"], c["chk"] or c["loc"], c["fam"])
            t[(c["fam"], "there" if pr else ("unsure" if un else "not there"), c["why"].split(" at ")[0][:26])] += 1
for k, v in sorted(t.items()):
    print(" ", k, v)

print("\nSTABLE IDENTITY. Rows seen by all 3 runs, and truth objects with one such row within", S.D, "px on some shared floor.")
for b in B:
    E = json.load(open(OUT / f"cells/{b}.json"))
    for F in ("stair", "shaft"):
        rs = [r for r in E["rows"] if r["fam"] == F]
        st = [r for r in rs if len(r["runs"]) == 3]
        obj = {o: v for o, v in T[b]["objects"].items() if fam(v["kind"]) == F}
        ok, part = 0, 0
        main_ok = main_n = 0
        for o, v in obj.items():
            def on(r):
                return any(box_dist(c["loc"], v["boxes"][f]) <= S.D for f, c in r["cells"].items() if f in v["boxes"] and c["loc"] and c["v"] > 0)
            any_ = [r for r in rs if on(r)]
            all3 = [r for r in st if on(r)]
            ok += bool(all3)
            part += bool(any_) and not all3
            if v["cls"] == "main":
                main_n += 1
                main_ok += bool(all3)
        extra = sum(1 for r in st if not any(box_dist(c["loc"], v["boxes"][f]) <= S.D for v in obj.values() for f, c in r["cells"].items() if f in v["boxes"] and c["loc"] and c["v"] > 0))
        print(f" {b:26s} {F:5s} rows {len(rs)} all-3-runs {len(st)} | truth {len(obj)}: stable {ok}, partial {part}, unseen {len(obj) - ok - part} | main {main_ok}/{main_n} | stable rows on nothing {extra}")

print("\nMARKERS. Spark's own point for each present claim, against the nearest true stair drawn on that floor. Claims more than 120 px from any are counted apart.")
for F in ("stair", "shaft"):
    al = []
    for b in B:
        pts, _, _ = C.points(b, b)
        w = []
        far = 0
        for p in pts:
            if p["fam"] != F:
                continue
            ds = [(box_dist(p["raw"], v["boxes"][str(p["floor"])]), float(np.hypot(*(np.array(p["raw"]) - np.array(ctr(v["boxes"][str(p["floor"])]))))), v["boxes"][str(p["floor"])]) for v in T[b]["objects"].values() if fam(v["kind"]) == F and str(p["floor"]) in v["boxes"]]
            if not ds or min(d[0] for d in ds) > 120:
                far += 1
                continue
            d = min(ds)
            w.append((d[0], d[1], (d[2][2] - d[2][0] + d[2][3] - d[2][1]) / 2))
        al += w
        if w:
            a = np.array(w)
            print(f" {F:5s} {b:26s} n {len(w)} far {far} inside box {np.mean(a[:, 0] == 0):.2f} to box median {np.median(a[:, 0]):.0f} p90 {np.percentile(a[:, 0], 90):.0f} max {a[:, 0].max():.0f} | to centre median {np.median(a[:, 1]):.0f} p90 {np.percentile(a[:, 1], 90):.0f}")
    if al:
        a = np.array(al)
        print(f" {F:5s} ALL n {len(a)} inside box {np.mean(a[:, 0] == 0):.2f} within 20 px {np.mean(a[:, 0] <= 20):.2f} within 40 px {np.mean(a[:, 0] <= 40):.2f} to box median {np.median(a[:, 0]):.0f} p90 {np.percentile(a[:, 0], 90):.0f} max {a[:, 0].max():.0f} | to centre median {np.median(a[:, 1]):.0f} p90 {np.percentile(a[:, 1], 90):.0f}; median box side {np.median(a[:, 2]):.0f}")
print(" row marker on Found cells, arm C, median of 3 runs, stairs")
al = []
for b in B:
    E = json.load(open(OUT / f"cells/{b}.json"))
    for c in rule.apply_rule(E, "C"):
        if c["state"] == "Found" and c["fam"] == "stair":
            ds = [box_dist(c["loc"], v["boxes"][str(c["f"])]) for v in T[b]["objects"].values() if fam(v["kind"]) == "stair" and str(c["f"]) in v["boxes"]]
            dk = [box_dist(c["chk"], v["boxes"][str(c["f"])]) for v in T[b]["objects"].values() if fam(v["kind"]) == "stair" and str(c["f"]) in v["boxes"]]
            if ds:
                al.append((min(ds), min(dk)))
a = np.array(al)
print(f"  n {len(a)} marker inside box {np.mean(a[:, 0] == 0):.2f} median {np.median(a[:, 0]):.0f} p90 {np.percentile(a[:, 0], 90):.0f} max {a[:, 0].max():.0f} | registered row place inside box {np.mean(a[:, 1] == 0):.2f} median {np.median(a[:, 1]):.0f} p90 {np.percentile(a[:, 1], 90):.0f} max {a[:, 1].max():.0f}")

print("\nCOST per building, clean stack. Sheet reads + 3 comparisons + crops. Seconds: 3 runs in parallel plus crops at 6 workers. Reused reads are priced as if paid.")
cost = {}
for b in B:
    E = json.load(open(OUT / f"cells/{b}.json"))
    s_usd, s_sec = 0, 0
    s = pres.load_spark()
    rr = [s[f"{b}/{r['i']}"]["info"] for r in pres.INDEX[b]] + [s[f"joint/{b}"]["info"]]
    s_usd = sum(i["usd"] for i in rr)
    s_sec = max(i["seconds"] for i in rr[:-1]) * np.ceil(len(rr[:-1]) / 5) + rr[-1]["seconds"]
    n = sum(1 for r in E["rows"] for c in r["cells"].values() if "crop" in c)
    nv = sum(1 for r in E["rows"] for c in r["cells"].values() if "crop" in c and c["v"] < 3)
    per = E["crop_usd"] / max(n, 1)
    secper = E["crop_seconds"] / max(n, 1)
    cost[b] = dict(sheet=round(s_usd, 3), compare=round(E["compare_usd"], 3), crops_all=n, crops_spec=nv, usd_spec=round(s_usd + E["compare_usd"] + nv * per, 3), usd_all=round(s_usd + E["compare_usd"] + n * per, 3),
                   usd_run0=round(s_usd + E["compare_usd"] / 3, 3), s_spec=round(s_sec + max(E["compare_seconds"]) + nv * secper / 6), s_all=round(s_sec + max(E["compare_seconds"]) + n * secper / 6))
    print(" ", b, cost[b])
print(" mean", {k: round(float(np.mean([c[k] for c in cost.values()])), 3) for k in next(iter(cost.values()))})
json.dump(dict(rows=rows_, cost=cost, tab={str(k): dict(v) for k, v in tab.items()}, detail=detail), open(OUT / f"report_{split}.json", "w"), indent=1, default=float)
