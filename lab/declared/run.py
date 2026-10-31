"""Runs every check on a scope and writes the flags. Usage: run.py dev|held [tag]. No model calls: Spark reads come from the cache."""
import json
import re
import sys

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/declared")
import math  # noqa: E402

import checks as C  # noqa: E402
import read_spark as RS  # noqa: E402
import read_tables as RT  # noqa: E402
import read_text as T  # noqa: E402
import rules as R  # noqa: E402


def pairs_for(sid, tbs, tabs):
    """Pair each door or window schedule with its plan page: the same page when it has tags, else the plan with the same sheet letter."""
    d = T.doc(sid)
    tagged = {p: RT.shape_tags(d[p]) for p in {p for p, _ in tabs}}
    out = []
    for p, tb in tabs:
        if tb["kind"] == "finish":
            continue
        marks = {r[tb["cols"][0]] for r in tb["rows"]}
        if tagged[p] and "PLAN" in tbs[p]["title"].upper():  # a plan on the same sheet owns its schedule, even when the numbers disagree
            out.append((p, tb, p))
            continue
        same = [t["page"] for t in tbs if t["sheet_no"][:1] == tbs[p]["sheet_no"][:1] and "PLAN" in t["title"].upper() and t["page"] != p]
        tagged.update({q: RT.shape_tags(d[q]) for q in same if q not in tagged})
        best = max(same, key=lambda q: len({t[0] for t in tagged[q]} & marks), default=None)
        if best is not None and {t[0] for t in tagged[best]} & marks:
            out.append((p, tb, best))
    return out


def run_text(sid, index_page=0):
    tbs = T.title_blocks(sid)
    C.check1(sid, tbs, T.cover_index(sid, index_page), cover_page=index_page)
    C.check6(sid, tbs)
    d = T.doc(sid)
    sheet_of = lambda p: tbs[p]["sheet_no"] or str(p)
    tabs = [(p, t) for p in range(len(d)) for t in RT.schedules(d[p])]
    C.check5(sid, sheet_of, pairs_for(sid, tbs, tabs))
    finish = [(p, t) for p, t in tabs if t["kind"] == "finish"]
    for p, t in finish:
        key, name = t["cols"][0], t["cols"][1]
        rows = [dict(no=r[key], name=r.get(name, ""), box=r["_box"], page=p) for r in t["rows"]]
        ex = [x["box"] for q, x in tabs if q == p]
        if "PLAN" in tbs[p]["title"].upper():
            plans = [(p, ex)]
        else:  # the schedule has its own sheet, so its plans are the plan sheets with the same letter
            plans = [(q["page"], []) for q in tbs if q["sheet_no"][:1] == tbs[p]["sheet_no"][:1] and "PLAN" in q["title"].upper()]
        C.check7(sid, sheet_of, rows, plans)
    return tbs


def reads(key):
    """Both runs of one Spark read, or None when a run is missing."""
    got = RS.load(key)
    a, b = got.get((key, 0)), got.get((key, 1))
    return (a, b) if a and b else None


def code(sid, page):
    return reads(f"code/{sid}/{page}")


def merged(prefix, field):
    """Rows of every tile of a read, per run."""
    got = RS.load(prefix)
    return [[x for k, v in sorted(got.items()) if k[1] == r and (k[0] == prefix or k[0].startswith(prefix + "/")) for x in v[field]] for r in (0, 1)]


def floor_levels(titles):
    return sorted({f for t in titles for f in R.floors(t) if re.search(r"FLOOR PLAN", t, re.I)})


def run_mml(sid):
    tbs = run_text(sid)
    r = code(sid, 0)
    C.stat(sid, 4, items=C.check4_area_table(sid))
    declared = None
    ls = T.lines(T.doc(sid)[0])
    lv = {x["t"].split()[0].rstrip(":").upper() for x in ls if re.match(r"^(Ground|Main|Upper)\s+Level", x["t"])}
    if lv:
        declared = sorted(lv)
    drawn = sorted({w for t in tbs for w in ("GROUND", "MAIN", "UPPER") if w in T_label(sid, t)})
    if r:
        C.check3(sid, r, "A1.0", 0, declared, drawn if declared else None)
    C.check2(sid, [t["title"] for t in tbs], False)


def T_label(sid, t):
    """Floor words on a floor plan sheet: title block and drawing labels."""
    if "PLAN" not in t["title"].upper() or "FLOOR PLAN" not in t["title"].upper() and not T.drawing_labels(T.doc(sid)[t["page"]]):
        return ""
    if re.search(r"PLUMBING|HVAC|LIGHTING|POWER|ROOF|FOUNDATION", t["title"], re.I):
        return ""
    return (t["title"] + " " + " ".join(x["t"] for x in T.drawing_labels(T.doc(sid)[t["page"]]))).upper()


def run_adu(sid):
    tbs = run_text(sid)
    r = code(sid, 0)
    n = C.check4_light_vent(sid, tbs[7]["sheet_no"])
    C.stat(sid, 4, items=n)
    if r:
        C.check3(sid, r, "G-001", 0)
    C.check2(sid, [t["title"] for t in tbs], True)


def run_af(sid):
    tbs = T.title_blocks(sid)
    C.check1(sid, tbs, T.cover_index(sid))
    C.check6(sid, tbs)
    C.check7_keys(sid)
    C.check2(sid, [t["title"] for t in tbs], False)
    C.stat(sid, 3, items=0, note="no code data on any sheet")
    C.stat(sid, 4, items=0, note="module totals only, no table to recompute")
    C.stat(sid, 5, items=0, note="no door schedule")
    C.stat(sid, 8, items=0, note="no life safety sheet")


def run_wfu(sid):
    C.check6_numbering(sid)
    C.check2(sid, [x["t"] for pg in T.doc(sid) for x in T.lines(pg) if x["size"] >= 15], False)
    for k, note in ((1, "no index and no title block"), (3, "no code data"), (4, "no tables"), (5, "no door tags or schedule"), (7, "no room schedule"), (8, "no life safety sheet")):
        C.stat(sid, k, items=0, note=note)


def run_usc():
    sid = "usc_leconte"
    tbs = T.title_blocks(sid)
    W, H = 3024, 2160
    for t in tbs[11:]:
        r = reads(f"tb/usc/{t['page']}")
        if r and R.norm(r[0]["sheet_number"]) == R.norm(r[1]["sheet_number"]):
            t.update(sheet_no=r[0]["sheet_number"], title=r[0]["sheet_title"], box=[0.88 * W, 0.86 * H, W, H])
    a, b = merged("index/usc", "rows")
    same = {x["sheet_number"]: x["title"] for x in a}
    other = {x["sheet_number"]: x["title"] for x in b}
    agreed = C.agreed(same, other)
    box = [0.05 * W, 0.72 * H, 0.9 * W, H]
    C.check1(sid, tbs, [dict(sheet_no=k, title=v, box=box) for k, v in agreed.items()], "spark")
    C.check6(sid, tbs)
    sheet_of = lambda p: tbs[p]["sheet_no"]
    d1, d2 = [{x["door"]: (x["width"], x["fire_rating"], x["hardware"]) for x in q} for q in merged("doors/usc", "rows")]
    rows = C.agreed(d1, d2)
    t1, t2 = [{x["tag"] for x in q} for q in merged("tags/usc", "tags")]
    h1, h2 = [{d: s["set"] for s in q for d in s["doors"]} for q in merged("hw/usc", "sets")]
    C.check5_spark(sid, "A400", rows, t1 & t2, C.agreed(h1, h2), 7, 4)
    f1, f2 = [{x["room_no"]: x["room_name"] for x in q} for q in merged("finish/usc", "rows")]
    C.check7(sid, sheet_of, [dict(no=k, name=v, page=9, sheet="A700", box=[0.35 * W, 0.69 * H, 0.75 * W, 0.985 * H]) for k, v in C.agreed(f1, f2).items()], [(4, [])], "spark", {t["sheet_no"] for t in tbs})
    r = code(sid, 1)
    C.check3(sid, r, "G002", 1)
    o = reads("occ/usc")
    if o:
        n = 0
        a, b = o[0]["rows"], o[1]["rows"]
        for x, y in zip(a, b):
            n += 1
            area, fac, occ = C.num(x["area"]), C.num(x["factor"]), C.num(x["occupants"])
            if (area, fac, occ) == (C.num(y["area"]), C.num(y["factor"]), C.num(y["occupants"])) and fac and R.occupants(area, fac) != occ:
                C.flag(sid, 4, "occupant_calc", "G002", f"{area:.0f} / {fac:.0f} = {area / fac:.2f}, rounds up to {R.occupants(area, fac)}, table says {occ:.0f}", [(1, [0.01 * W, 0.67 * H, 0.235 * W, 0.78 * H])], "spark")
        C.stat(sid, 4, items=n)
    e = [[(x["x"], x["y"], x["width"], x["capacity"]) for x in q] for q in tile_exits("exits/usc", sid, 2)]
    C.check8(sid, "G003", 2, e)
    C.check2(sid, [t["title"] for t in tbs], True)


def tile_exits(prefix, sid, page):
    """Exit markers per run with tile coordinates mapped to 0-1000 of the sheet."""
    clips = RS.tiles(sid, page)
    got = RS.load(prefix)
    out = [[], []]
    for (key, run), v in got.items():
        i = int(key.rsplit("/", 1)[1])
        c = clips[i]
        for m in v["exits"]:
            out[run].append(dict(m, x=1000 * (c[0] + m["x"] / 1000 * (c[2] - c[0])), y=1000 * (c[1] + m["y"] / 1000 * (c[3] - c[1]))))
    return out


def run_brandeis():
    sid = "brandeis_hall"
    ls = T.lines(T.doc(sid)[0])
    lab = [x for x in ls if re.search(r"LEVEL", x["t"])]
    fl = [R.floors(x["t"]) for x in lab]
    C.stat(sid, 6, items=len(lab), pages=1)
    if len({frozenset(f) for f in fl if f}) > 1:
        C.flag(sid, 6, "floor_label", "1", f"labels disagree: {[x['t'] for x in lab]}", [(0, [x["x0"], x["y0"], x["x1"], x["y1"]]) for x in lab[:3]])
    C.check2(sid, [x["t"] for x in ls if x["size"] >= 20], False)
    for k, note in ((1, "one sheet, no index"), (3, "no code data"), (4, "no tables"), (5, "no door tags or schedule"), (7, "no room schedule"), (8, "no life safety data")):
        C.stat(sid, k, items=0, note=note)


def ifc_spaces():
    """Room number and name of every IfcSpace in the clinic model. No ifcopenshell: the STEP line holds both."""
    text = (T.ROOT / "data/plans/nibs_clinic/Clinic_A_20110906_optimized.ifc").read_text(errors="replace")
    return dict(re.findall(r"IFCSPACE\('[^']*',#\d+,'([^']*)','[^']*',[^,]*,#\d+,#\d+,'([^']*)'", text))


def door_room(no):
    """Door 1A15A or 1AC5a opens from room 1A15 or 1AC5."""
    return re.sub(r"[A-Za-z]$", "", no) if re.search(r"\d[A-Za-z]$", no) else no


def run_clinic():
    sid = "nibs_clinic"
    files = RS.clinic_files()
    # index rows, two reads per crop, keep what both reads agree on
    a, b = merged("index/clinic", "rows")
    ra = {x["sheet_number"]: x for x in a if x["sheet_number"] not in ("", "-")}
    rb = {x["sheet_number"]: x for x in b if x["sheet_number"] not in ("", "-")}
    both = {k: ra[k] for k in ra if k in rb and R.title_same(ra[k]["title"], rb[k]["title"])}
    C.stat(sid, 1, index_read=[len(ra), len(rb)], index_agreed=len(both))
    tbf = RS.load("tbc/")
    tbs = []
    for f in files:
        r0, r1 = tbf.get((f"tbc/{f}", 0)), tbf.get((f"tbc/{f}", 1))
        r = r0 or {}
        tbs.append(dict(page=f, sheet_no=r.get("sheet_number", ""), title=r.get("sheet_title", ""), box=[0.78 * 3024, 0.84 * 2016, 3024, 2016], second=r1))
    # a flag needs the second read to say the same thing, so the first pass only names the files to read again
    want = [t["page"] for t in tbs if t["second"] is None and (t["sheet_no"] not in both or not R.title_same(both[t["sheet_no"]]["title"], t["title"]))]
    (C.OUT / "clinic_recheck.json").write_text(json.dumps(want))
    keep = []
    for t in tbs:
        s = t["second"]
        if s and R.norm(s["sheet_number"]) != R.norm(t["sheet_no"]):  # the two reads disagree on the number, so this sheet stays unread
            continue
        keep.append(t)
    rect = T.page(sid, RS.CL["index"]).rect
    C.check1(sid, keep, [dict(sheet_no=k, title=v["title"], box=[0, 0, rect.width, rect.height]) for k, v in both.items()], "spark", cover_page=RS.CL["index"])
    # code notes on the life safety sheets
    for k, sh in (("c161", "A18-1"), ("c162", "A18-2")):
        r = code(sid, RS.CL[k])
        if r:
            C.check3(sid, r, sh, RS.CL[k])
    # floor label: title block against the drawing title
    n = 0
    for k, sh in (("c161", "A18-1"), ("c162", "A18-2")):
        r = reads(f"label/clinic/{k}")
        if r:
            n += 1
            ft, fl = R.floors(r[0]["sheet_title"]), R.floors(r[0]["drawing_title"])
            if ft and fl and not ft & fl and R.floors(r[1]["sheet_title"]) == ft:
                pr = T.page(sid, RS.CL[k]).rect
                C.flag(sid, 6, "floor_label", sh, f"title block {r[0]['sheet_title']!r}, drawing {r[0]['drawing_title']!r}", [(RS.CL[k], [0.75 * pr.width, 0.6 * pr.height, pr.width, pr.height])], "spark")
    C.stat(sid, 6, items=n, pages=len(files))
    # schedules
    def rows_of(prefix, field, key):
        x, y = merged(prefix, field)
        return [{R.norm(q[key]).replace(" ", ""): q for q in z if q[key].strip()} for z in (x, y)]
    d1, d2 = rows_of("doors/clinic", "rows", "door")
    f1, f2 = rows_of("finish/clinic", "rows", "room_no")
    doors = {k: v for k, v in d1.items() if k in d2 and R.norm(v["room"]) == R.norm(d2[k]["room"])}
    finish = {k: v for k, v in f1.items() if k in f2 and R.norm(v["room_name"]) == R.norm(f2[k]["room_name"])}
    C.stat(sid, 5, items=len(doors), read=[len(d1), len(d2)], note="door tags on plans not read: scope is sheets 161, 162 and schedules")
    sched = T.page(sid, RS.CL["d79"]).rect
    box = [0.06 * sched.width, 0, 0.73 * sched.width, sched.height]
    for k, v in doors.items():
        rm = door_room(k)
        if rm not in finish:
            C.flag(sid, 7, "door_room_not_in_finish", "A5", f"door {k} opens from room {rm} {v['room']!r}, which has no finish schedule row", [(RS.CL["d79"], box)], "spark")
        elif not C.room_same(v["room"], finish[rm]["room_name"]):
            C.flag(sid, 7, "room_name_mismatch", "A5", f"door {k}: door schedule room {v['room']!r}, finish schedule {rm} {finish[rm]['room_name']!r}", [(RS.CL["d79"], box)], "spark")
    live = {k: v for k, v in finish.items() if "NOT USED" not in R.norm(v["room_name"]) and "NOTUSED" not in R.norm(v["room_name"]).replace(" ", "")}
    model = {R.norm(k).replace(" ", ""): v for k, v in ifc_spaces().items() if re.match(r"^[12][A-Z]", k) and "ROOF" not in v.upper()}
    for k, v in live.items():
        if k not in model:
            C.flag(sid, 7, "schedule_room_not_on_plan", "A4", f"finish schedule {k} {v['room_name']!r} is not in the model", [(RS.CL["f76"], [0.235 * sched.width, 0, 0.685 * sched.width, sched.height])], "spark")
        elif not C.room_same(v["room_name"], model[k]):
            C.flag(sid, 7, "room_name_mismatch", "A4", f"room {k}: finish schedule {v['room_name']!r}, model {model[k]!r}", [(RS.CL["f76"], [0.235 * sched.width, 0, 0.685 * sched.width, sched.height])], "spark")
    for k, v in model.items():
        if k not in finish:
            C.flag(sid, 7, "plan_room_not_in_schedule", "A4", f"model room {k} {v!r} has no finish schedule row", [(RS.CL["f76"], [0.235 * sched.width, 0, 0.685 * sched.width, sched.height])], "spark")
    C.stat(sid, 7, items=len(live), model_rooms=len(model), door_rows_tested=len(doors), read=[len(f1), len(f2)])
    # exit markers. The stair boxes on 162 print a capacity and no width, so no width check runs there.
    exits = {}
    for k, sh, per in (("c161", "A18-1", R.EXIT_IN_PER_OCC), ("c162", "A18-2", R.STAIR_IN_PER_OCC)):
        e = [[(x["x"], x["y"], x["width"], x["capacity"]) for x in q] for q in tile_exits(f"exits/clinic/{k}", sid, RS.CL[k])]
        both_, tot = C.check8(sid, sh, RS.CL[k], e, factor=per)
        exits[k] = (len(both_), tot)
    C.stat(sid, 8, per_sheet=exits)
    C.stat(sid, 2, code_notes=True)
    C.check2(sid, [re.sub(r"^Clinic_\d+_", "", f[:-4]) for f in files], True)
    # travel distances printed in feet and metres
    n = 0
    for k, sh in (("c161", "A18-1"), ("c162", "A18-2")):
        vals = {v for q in RS.load(f"exits/clinic/{k}").values() for v in q["travel_distances"]}
        for v in vals:
            m = re.search(r"(\d+(?:\.\d+)?)\s*FT\s*/\s*(\d+(?:\.\d+)?)\s*M", v.upper())
            if m:
                n += 1
                ft, me = float(m[1]), float(m[2])
                if abs(ft * 0.3048 - me) > R.FT_TO_M_TOL:
                    C.flag(sid, 4, "unit_conversion", sh, f"{v}: {ft} ft is {ft * 0.3048:.2f} m", [(RS.CL[k], [0, 0, 3024, 2016])], "spark")
    C.stat(sid, 4, items=n, note="no occupant table in the set scope")


def run_bcoe():
    sid = "bcoe_addendum3"
    C.check4_bcoe()
    r = code(sid, 1)
    if r:
        C.check3(sid, r, "A012", 1)
    pg = T.doc(sid)[1]
    ls = T.lines(pg)
    wd = pg.get_text("words")  # words, because a revision triangle's number can share a line with the count
    d = json.loads((T.ROOT / "data/lab/gather_practice/declared_a012.json").read_text())
    n = 0
    for e in d["exits"]:
        lab = [x for x in ls if x["t"] == f"EXIT {e['exit']}" and x["x0"] < 1980]
        if lab:
            near = [w[4] for w in wd if re.fullmatch(r"\d{2,3}", w[4]) and math.dist((w[0], w[1]), (lab[0]["x0"], lab[0]["y0"])) < 130 and w[0] < 1980]
            n += 1
            vals = {int(o) for o in near}
            if e["occupants"] not in vals:
                C.flag(sid, 8, "exit_label_vs_table", "A012", f"exit {e['exit']}: table {e['occupants']} occupants, plan numbers near the exit {sorted(vals)}", [(1, [lab[0]["x0"] - 70, lab[0]["y0"] - 70, lab[0]["x1"] + 70, lab[0]["y1"] + 90]), (1, [1990, 1840, 2420, 1925])])
    C.stat(sid, 8, items=n, capacity_sum=d["exit_occupant_sum"])
    # the occupancy table is the set's room list
    words = pg.get_text("words")
    import declared_a012 as DA
    rows = [m.groups() for m in map(DA.ROOM.match, DA.merge_split_rows(DA.rows(words, 1990, 2420, 300, 970))) if m]
    C.check7(sid, lambda p: "A012", [dict(no=x[0], name=x[1], page=1, sheet="A012", box=[1990, 300, 2420, 970]) for x in rows], [(1, [[1980, 0, 3024, 2160]])])
    for k, note in ((1, "only 3 of the set's sheets are in the addendum"), (2, "only 3 of the set's sheets are in the addendum"), (5, "door schedule A111 present, no door tags on the 3 sheets"), (6, "no floor label pair")):
        C.stat(sid, k, items=0, note=note)


def main(scope, tag):
    sets = R.DEV if scope == "dev" else R.HELD
    for sid in sets:
        if sid.startswith("mml_"):
            run_mml(sid)
        elif sid.startswith("carpinteria"):
            run_adu(sid)
        elif sid.startswith("af_ces"):
            run_af(sid)
        elif sid.startswith("wfu"):
            run_wfu(sid)
        elif sid == "usc_leconte":
            run_usc()
        elif sid == "bcoe_addendum3":
            run_bcoe()
        elif sid == "brandeis_hall":
            run_brandeis()
        elif sid == "nibs_clinic":
            run_clinic()
    out = C.OUT / f"flags_{scope}{tag}.json"
    out.write_text(json.dumps(dict(version=R.VERSION, flags=C.FLAGS, stats=C.STATS), indent=1))
    by = {}
    for f in C.FLAGS:
        by.setdefault((f["set"], f["check"], f["sub"]), []).append(f)
    for k in sorted(by, key=str):
        print(k, len(by[k]))
        for f in by[k]:
            print("    ", f["sheet"], f["text"][:150])
    print(json.dumps(C.STATS, indent=None, default=str)[:1800])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
