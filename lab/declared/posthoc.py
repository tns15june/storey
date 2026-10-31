"""Fixes made after the held-out flags were seen. Everything here is post hoc and is scored apart from the frozen v2 result.

The frozen modules (rules.py, checks.py, read_*.py, run.py) stay as stamped. New Spark reads use keys that start with v3.
Usage: posthoc.py strips | read | flags
"""
import json
import math
import re
import sys

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/declared")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import checks as C  # noqa: E402
import read_spark as RS  # noqa: E402
import read_text as T  # noqa: E402
import rules as R  # noqa: E402
import run as RN  # noqa: E402

SID = "nibs_clinic"
OUT = C.OUT
DOOR_CLIP = {"d79": (0.06, 0.005, 0.73, 0.985), "d80": (0.06, 0.005, 0.73, 0.985), "d81": (0.015, 0.03, 0.70, 0.985)}
PLAN_CLIP = (0.27, 0.10, 0.70, 0.74)
ROOM_P = """This image is part of a floor plan. Room labels are a room name with a room number in a small box under it, such as CORRIDOR over 1EC1. List every room label you can read. number: the number in the box, exactly as printed. name: the name above it, exactly as printed, with line breaks joined. x, y: the centre of the number in the 0-1000 range of this image. Skip labels cut by the image edge. Do not list device symbols such as F or FEC."""
ROOM_S = RS.obj(rooms=RS.arr(RS.obj(number=RS.STR, name=RS.STR, x=RS.INT, y=RS.INT)))
JUDGE_P = """Two strings name one sheet of a building drawing set. "index" comes from the sheet index on the cover, "block" from the title block on the sheet itself. For each pair decide if they can name the same drawing. Answer same when one is an abbreviation, a longer or shorter form, a synonym, a spelling variant or a typo of the other. Answer different when they name a different floor, block, building part, trade or subject, even if most words match. Return one result per id.

"""
JUDGE_S = RS.obj(results=RS.arr(RS.obj(id=RS.STR, answer={"type": "string", "enum": ["same", "different"]})))
MATERIAL = {"STEEL", "WOOD", "ALUM", "ALUMINUM", "HM", "VINYL", "GLASS", "FIBERGLASS"}


def rules_y(im, frac=0.8):
    """y of the ruled lines that cross most of a table."""
    ink = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 170).mean(1)
    idx = np.where(ink > frac)[0]
    return [int(g.mean()) for g in np.split(idx, np.where(np.diff(idx) > 2)[0] + 1) if len(g)]


def row_strips(sid, f, clip, n=4, px=2000):
    """Strips cut on the table's own row lines, each row once, under a header that stops at its bottom rule."""
    im = RS.render(sid, f, 6000, clip)
    ys = rules_y(im)
    gaps = np.diff(ys)
    h = float(np.median(gaps))
    start = next(i for i in range(len(gaps) - 5) if all(abs(g - h) <= 3 for g in gaps[i:i + 5]))
    body = ys[start:]
    assert len(body) > 20 and all(g < 2.5 * h for g in np.diff(body)), (f, len(body))
    head = im[max(0, ys[0] - 4):body[0] + 3]
    cut = np.linspace(0, len(body) - 1, n + 1, dtype=int)
    out = []
    for i in range(n):
        s = np.vstack([head, im[body[cut[i]] + 2:body[cut[i + 1]] + 3]])
        k = px / max(s.shape[:2])
        out.append(cv2.resize(s, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else s)
    return out, len(body) - 1


def sub_tiles(clip, cols=3, rows=3, overlap=0.08):
    x0, y0, x1, y1 = clip
    w, h = (x1 - x0), (y1 - y0)
    out = []
    for c in RS.tiles(None, None, cols, rows, overlap):
        out.append((x0 + c[0] * w, y0 + c[1] * h, x0 + c[2] * w, y0 + c[3] * h))
    return out


def jobs():
    j = []
    for k in ("d79", "d80", "d81"):
        for i in range(4):
            for r in (0, 1):
                j.append(lambda k=k, i=i, r=r: RS.ask(f"v3doors/{k}/{i}", r, RS.DOOR_P + " The image shows part of the table with its header row on top.", lambda: row_strips(SID, RS.CL[k], DOOR_CLIP[k])[0][i], RS.DOOR_S, "medium"))
    for k in ("c161", "c162"):
        for i, c in enumerate(sub_tiles(PLAN_CLIP)):
            for r in (0, 1):
                j.append(lambda k=k, i=i, c=c, r=r: RS.ask(f"v3rooms/{k}/{i}", r, ROOM_P, lambda: RS.render(SID, RS.CL[k], 2000, c), ROOM_S, "medium", 8000))
    return j


def judge_pairs():
    """Every title flag from v2, with the project name taken off the title block string."""
    out = []
    for name in ("flags_dev_v2.json", "flags_held_v2.json"):
        for f in json.load(open(OUT / name))["flags"]:
            if f["sub"] == "title_mismatch":
                m = re.search(r"index (.+) against title block (.+)$", f["text"])
                a, b = eval_str(m[1]), eval_str(m[2])
                out.append(dict(id=f["id"], index=a, block=re.sub(r"^THE \w+ \(Module \w\)\s*", "", b)))
    return out


def eval_str(s):
    return s.strip()[1:-1]


def run_judge():
    import metalab as m
    pairs = judge_pairs()
    prompt = JUDGE_P + json.dumps(pairs, indent=0)
    for r in (0, 1):
        k = ("v3judge", r)
        if k in RS._done:
            continue
        out, info = m.spark(prompt, schema=JUDGE_S, effort="low", max_tokens=6000)
        row = dict(key="v3judge", run=r, effort="low", out=out, usd=info["usd"], seconds=info["seconds"], input=info["input"], output=info["output"])
        RS._done[k] = row
        with open(RS.CACHE, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "strips":
        for k in DOOR_CLIP:
            s, n = row_strips(SID, RS.CL[k], DOOR_CLIP[k])
            print(k, n, [x.shape for x in s])
            for i, x in enumerate(s):
                cv2.imwrite(sys.argv[2] + f"/v3_{k}_{i}.png", x)
    if what == "read":
        RS.pool(jobs())
        run_judge()
        print(sum(1 for k in RS._done if k[0].startswith("v3")), "v3 reads", round(sum(v["usd"] for k, v in RS._done.items() if k[0].startswith("v3")), 3), "usd")


def door_rows():
    """v3 door rows per run: {door: row}, seam-free strips."""
    out = []
    for r in (0, 1):
        d = {}
        for k in ("d79", "d80", "d81"):
            for i in range(4):
                got = RS._done.get((f"v3doors/{k}/{i}", r))
                for x in (got["out"]["rows"] if got else []):
                    d.setdefault(R.norm(x["door"]).replace(" ", ""), x)
        out.append(d)
    return out


def plan_rooms():
    """v3 room labels per sheet and run, with sheet coordinates in points."""
    out = {}
    for k in ("c161", "c162"):
        clips = sub_tiles(PLAN_CLIP)
        rect = T.page(SID, RS.CL[k]).rect
        out[k] = [{}, {}]
        for i, c in enumerate(clips):
            for r in (0, 1):
                got = RS._done.get((f"v3rooms/{k}/{i}", r))
                for x in (got["out"]["rooms"] if got else []):
                    n = R.norm(x["number"]).replace(" ", "").replace("-", "")
                    px = (c[0] + x["x"] / 1000 * (c[2] - c[0])) * rect.width
                    py = (c[1] + x["y"] / 1000 * (c[3] - c[1])) * rect.height
                    if n:
                        out[k][r].setdefault(n, dict(number=n, name=x["name"], x=px, y=py))
    return out


def core_exits(k):
    """Exit markers with a printed capacity, each kept only from the tile whose half of the sheet holds it, so an overlap cannot double one."""
    clips = RS.tiles(SID, RS.CL[k])
    out = [[], []]
    for (key, run), v in RS.load(f"exits/clinic/{k}").items():
        i = int(key.rsplit("/", 1)[1])
        c = clips[i]
        for m in v["exits"]:
            x, y = c[0] + m["x"] / 1000 * (c[2] - c[0]), c[1] + m["y"] / 1000 * (c[3] - c[1])
            if m["capacity"] and (x < 0.5) == (c[0] < 0.25) and (y < 0.5) == (c[1] < 0.25):
                out[run].append((round(x * 1000), round(y * 1000), m["width"], m["capacity"]))
    return out


if __name__ == "__main__" and sys.argv[1] == "look":
    truth = json.load(open(OUT / "truth.json"))
    want = {x.upper() for k in ("doors_079", "doors_080", "doors_081") for x in truth["nibs_clinic"][k]}
    a, b = door_rows()
    print("v3 door rows", len(want), len(a), len(b), "both", len(set(a) & set(b) & want), "extra", sorted((set(a) | set(b)) - want), "miss0", sorted(want - set(a))[:30], "miss1", sorted(want - set(b))[:30])
    sp = truth["nibs_clinic"]["door_specials"]
    ok = [k for k, (w, f) in sp.items() if k in a and R.norm(w).replace(" ", "") in R.norm(a[k]["width"]).replace(" ", "") and (not f or f in a[k]["fire_rating"])]
    print("specials", len(ok), len(sp), [k for k in sp if k not in ok])
    res = {r: {x["id"]: x["answer"] for x in RS._done[("v3judge", r)]["out"]["results"]} for r in (0, 1)}
    pairs = judge_pairs()
    sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/declared")
    import verdicts as V
    dev = json.load(open(OUT / "flags_dev_v2.json"))["flags"]
    held = json.load(open(OUT / "flags_held_v2.json"))["flags"]
    fl = {f["id"]: f for f in dev + held}
    for p in pairs:
        f = fl[p["id"]]
        v = V.DEV[p["id"]] if p["id"] in V.DEV else V.held_verdict(f)
        print(p["id"][:36].ljust(36), res[0].get(p["id"]), res[1].get(p["id"]), v[0].ljust(6), p["index"][:34], "|", p["block"][:40])
    for k in ("c161", "c162"):
        print(k, "core", [sum(x[3] for x in q) for q in core_exits(k)], core_exits(k))
    pr = plan_rooms()
    for k, (q0, q1) in pr.items():
        print(k, len(q0), len(q1), "both", len(set(q0) & set(q1)), "only0", sorted(set(q0) - set(q1))[:40], "only1", sorted(set(q1) - set(q0))[:40])


def finish_rows():
    a, b = RN.merged("finish/clinic", "rows")
    f0 = {R.norm(x["room_no"]).replace(" ", ""): x["room_name"] for x in a if x["room_no"].strip()}
    f1 = {R.norm(x["room_no"]).replace(" ", ""): x["room_name"] for x in b if x["room_no"].strip()}
    # a name that differs by one letter between reads is a printed typo read twice, so the row stays
    return {k: v for k, v in f0.items() if k in f1 and C.room_same(v, f1[k])}


def clinic7():
    """Plan labels on 161 and 162 and door rows against the finish schedule. Returns raw comparison lists."""
    fin = finish_rows()
    live = {k: v for k, v in fin.items() if "NOT USED" not in R.norm(v)}
    pr = plan_rooms()
    both = {}
    for k, (q0, q1) in pr.items():
        for n in set(q0) & set(q1):
            both.setdefault(n, dict(q0[n], sheet=k, name1=q1[n]["name"]))
    either = {n for k, (q0, q1) in pr.items() for n in set(q0) | set(q1)}
    out = dict(name=[], absent=[], ghost=[], door=[])
    for n, p in sorted(both.items()):
        if n not in fin:
            out["ghost"].append((n, p["name"], p["sheet"], p["x"], p["y"]))
        elif n in live and not (C.room_same(p["name"], live[n]) or C.room_same(p["name1"], live[n])):
            out["name"].append((n, p["name"], live[n], p["sheet"], p["x"], p["y"]))
    for n, v in sorted(live.items()):
        if n not in either:
            out["absent"].append((n, v))
    d0, d1 = door_rows()
    doors = {k: v for k, v in d0.items() if k in d1 and R.norm(v["room"]) == R.norm(d1[k]["room"])}
    for k, v in sorted(doors.items()):
        rm = k if k in fin else RN.door_room(k)
        if rm not in fin:
            out["door"].append((k, v["room"], rm, None))
        elif not C.room_same(v["room"], fin[rm]):
            out["door"].append((k, v["room"], rm, fin[rm]))
    return out, dict(finish_rows=len(fin), live=len(live), plan_labels=len(both), door_rows=len(doors))


if __name__ == "__main__" and sys.argv[1] == "c7":
    o, st = clinic7()
    print(st)
    for k, v in o.items():
        print(k, len(v))
        for x in v:
            print("   ", x)


KEY = re.compile(r"^[A-Z]\d{1,2}$")


def keys_v3(sid):
    """Legend keys and plan keys. A legend row can arrive as one text line, key and name together."""
    legend, plan = {}, {}
    for i, pg in enumerate(T.doc(sid)):
        ls = [x for x in T.lines(pg) if x["x0"] < 0.9 * pg.rect.width and x["size"] >= 9]
        for x in ls:
            m = re.match(r"^([A-Z]\d{1,2})\s{2,}(\S.*)$", x["t"])
            if m:
                legend[m[1]] = m[2]
            elif KEY.match(x["t"]):
                nm = [o for o in ls if 0 < o["x0"] - x["x1"] < 40 and abs(o["y0"] - x["y0"]) < 3 and re.search("[A-Z]{3}", o["t"])]
                if nm:
                    legend[x["t"]] = nm[0]["t"]
                else:
                    plan.setdefault(x["t"], i)
    return legend, plan


def v3_rows():
    """Post hoc flag list: frozen rows with a rule removing or keeping each, plus the new flags."""
    y = json.load(open(OUT / "yield_v2.json"))
    res = {r: {x["id"]: x["answer"] for x in RS._done[("v3judge", r)]["out"]["results"]} for r in (0, 1)}
    block = lambda s: set(re.findall(r"BLOCKS? \"?(\d[A-Z])", s.upper()))
    pairs = {p["id"]: p for p in judge_pairs()}
    keep, gone = [], []
    for r in y["rows"]:
        why = None
        if r["sub"] == "index_missing_sheet" and ("NOT USED" in r["text"] or r["sheet"] == "P3-2"):
            why = "NOT USED rows skipped; either title block read may match the index"
        elif r["sub"] == "title_mismatch":
            p = pairs[r["id"]]
            hard = (R.floors(p["index"]) and R.floors(p["block"]) and not R.floors(p["index"]) & R.floors(p["block"])) or (block(p["index"]) and block(p["block"]) and block(p["index"]) != block(p["block"]))
            if not hard and not (res[0][r["id"]] == "different" and res[1][r["id"]] == "different"):
                why = "Spark judge said same in a read, no floor or block conflict"
        elif r["sub"] == "schedule_cell" and not re.search(r"reads '(" + "|".join(MATERIAL) + ")'", r["text"]):
            why = "only a value from another column is a slip"
        elif r["sub"] == "marker_no_capacity":
            if r["set"] == "usc_leconte":
                r = dict(r, sub="no_capacity_on_sheet")
            else:
                why = "markers count only with a printed capacity"
        elif r["set"] == "nibs_clinic" and r["sub"] in ("door_room_not_in_finish", "room_name_mismatch", "plan_room_not_in_schedule", "schedule_room_not_on_plan"):
            why = "replaced by plan labels and row-cut door rows"
        elif r["sub"] == "plan_key_not_in_legend":
            why = "legend parser reads key and name from one line"
        (gone if why else keep).append(dict(r, why=why))
    o, st = clinic7()
    new = []
    add = lambda sub, check, sheet, text, verdict, root, note, source="spark", sid=SID: new.append(dict(id=f"v3.{sid}.{check}.{sub}.{len(new)}", set=sid, check=check, sub=sub, sheet=sheet, text=text, source=source, scope="held" if sid in R.HELD else "dev", verdict=verdict, root=root, note=note))
    for n, pn, fn, k, *_ in o["name"]:
        v = ("likely", "clinic-1e24", "plan and door schedule agree, finish differs") if n == "1E24" else ("real", "clinic-2b19", "plan OFFICE, finish and door schedule WATER LAB") if n == "2B19" else ("real", "clinic-1e25-28", "finish rows hold the wrong rooms")
        add("room_name_mismatch", 7, {"c161": "A18-1", "c162": "A18-2"}[k], f"room {n}: plan {pn!r}, finish schedule {fn!r}", *v)
    for d, dn, rm, fn in o["door"]:
        v = ("likely", "clinic-1e24", "plan and door schedule agree, finish differs") if rm == "1E24" else ("real", "clinic-1e25-28", "finish rows hold the wrong rooms")
        add("room_name_mismatch", 7, "A5", f"door {d}: door schedule {dn!r}, finish schedule {rm} {fn!r}", *v)
    add("plan_room_not_in_schedule", 7, "A18-1", "plan label read as 1CD1, no finish row", "false", "", "Spark label read: 1C01 read as 1CD1 in both reads")
    add("plan_room_not_in_schedule", 7, "A18-2", "plan label E1 ELEVATOR, no finish row", "false", "", "rule: an elevator cab has no finish row")
    sid = "bcoe_addendum3"
    add("assembly_load_in_B", 3, "A012", "room 106 FAMILY ROOM carries 54 occupants, the set declares B only", "likely", "bcoe-assembly", "rooms 106 and 144 total 2,411 SF of 11,538 SF, no second occupancy stated on the 3 sheets in the addendum", "text", sid)
    add("assembly_load_in_B", 3, "A012", "room 144 CONFERENCE ROOM carries 107 occupants, the set declares B only", "likely", "bcoe-assembly", "same", "text", sid)
    return keep + new, gone, o, st


def tally(rows):
    c = {k: sum(r["verdict"] == k for r in rows) for k in ("real", "likely", "false", "cannot_tell")}
    return dict(flags=sum(c.values()), **c, roots=len({r["root"] for r in rows if r["verdict"] in ("real", "likely") and r["root"]}))


if __name__ == "__main__" and sys.argv[1] == "flags":
    rows, gone, o, st = v3_rows()
    scored = [r for r in rows if r["verdict"] != "not_a_submission"]
    res = dict(phase="post hoc", by_scope={k: tally([r for r in scored if r["scope"] == k]) for k in ("dev", "held")})
    res["by_check"] = {k: tally([r for r in scored if r["check"] == k]) for k in range(1, 9)}
    res["by_sub"] = {f"{k}.{s}": tally([r for r in scored if r["sub"] == s]) for k, s in sorted({(r["check"], r["sub"]) for r in scored})}
    res["by_set"] = {s: tally([r for r in scored if r["set"] == s]) for s in R.DEV + R.HELD}
    res["lost_real_or_likely"] = [(r["id"], r["verdict"], r["why"]) for r in gone if r["verdict"] in ("real", "likely")]
    res["cannot_tell_rooms"] = [x[0] for x in o["absent"]]
    res["clinic7_stats"] = st
    res["exit_core"] = {k: [sum(x[3] for x in q) for q in core_exits(k)] for k in ("c161", "c162")}
    (OUT / "yield_v3.json").write_text(json.dumps(dict(res, rows=rows, gone=gone), indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_sub", "cannot_tell_rooms")}, default=str)[:2400])
    for k, v in res["by_sub"].items():
        print(k, v)
    for sid in ("af_ces_admin", "af_ces_shops"):
        lg, pl = keys_v3(sid)
        print(sid, len(lg), len(pl), "plan not legend", sorted(set(pl) - set(lg)), "legend not plan", sorted(set(lg) - set(pl)))
