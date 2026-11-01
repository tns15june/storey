"""Text-layer tables and plan tags: door, window and finish schedules, tags inside small shapes, room labels. No model."""
import collections
import math
import re
import sys

sys.path.insert(0, "lab/declared")
import fitz  # noqa: E402
import read_text as T  # noqa: E402

KINDS = {"door": re.compile(r"DOOR\s+(AND FRAME\s+)?SCHEDULE", re.I), "window": re.compile(r"WINDOW SCHEDULE", re.I), "finish": re.compile(r"FINISH SCHEDULE", re.I)}
FIRST = {"MARK", "NO.", "NUMBER", "NO"}


def rows_of(ls, tol=3.5):
    """Group lines into rows by y."""
    rows = []
    for x in sorted(ls, key=lambda a: a["y0"]):
        if rows and abs(rows[-1][0]["y0"] - x["y0"]) <= tol:
            rows[-1].append(x)
        else:
            rows.append([x])
    return [sorted(r, key=lambda a: a["x0"]) for r in rows]


def schedules(pg):
    """Each schedule on the page as dict(kind, title, cols, rows, box). Cells go to the nearest column centre."""
    ls = [x for x in T.lines(pg) if x["dir"] == (1, 0)]
    out, seen = [], set()
    for kind, rx in KINDS.items():
        for t in ls:
            if not (rx.search(t["t"]) and t["size"] >= 9 and len(t["t"]) < 60):
                continue
            near = [x for x in ls if t["y0"] < x["y0"] < t["y0"] + 95 and t["x0"] - 40 < x["x0"] < t["x0"] + 720 and x is not t]
            head = next((r for r in rows_of(near) if r[0]["t"].upper() in FIRST and len(r) >= 3), None)
            if head is None or (kind, round(head[0]["y0"])) in seen:
                continue
            seen.add((kind, round(head[0]["y0"])))
            head = [c for i, c in enumerate(head) if i == 0 or c["x0"] - head[i - 1]["x1"] < 150]  # a stray callout on the header row is not a column
            cols = [(c["t"].upper(), (c["x0"] + c["x1"]) / 2) for c in head]
            xmin, xmax = head[0]["x0"] - 30, head[-1]["x1"] + 130
            body = [x for x in ls if x["y0"] > head[0]["y0"] + 3 and xmin < x["x0"] < xmax and x["size"] <= head[0]["size"] + 1]
            rows, last = [], head[0]["y0"]
            for r in rows_of(body):
                if r[0]["y0"] - last > 45:
                    break
                row = {}
                for c in r:
                    name = min(cols, key=lambda k: abs(k[1] - (c["x0"] + c["x1"]) / 2))[0]
                    row[name] = (row.get(name, "") + " " + c["t"]).strip()
                key = cols[0][0]
                if key in row and re.fullmatch(r"[A-Z0-9.\-]{1,8}", row[key]):
                    row["_box"] = [min(c["x0"] for c in r), r[0]["y0"], max(c["x1"] for c in r), max(c["y1"] for c in r)]
                    rows.append(row)
                    last = r[0]["y0"]
                elif rows and len(r) >= 2:
                    break
            if rows:
                box = [min(x["x0"] for x in head + [t]), t["y0"], max(x["x1"] for x in head), rows[-1]["_box"][3]]
                out.append(dict(kind=kind, title=t["t"], cols=[c[0] for c in cols], rows=rows, box=box))
    return out


def shape_tags(pg):
    """Short words inside a small closed shape. Returns (word, cx, cy, family). family is (w, h, kind) of the shape in the rendered frame."""
    M = pg.rotation_matrix
    shapes = []
    for d in pg.get_drawings():
        r = d["rect"]
        if 8 <= r.width <= 32 and 8 <= r.height <= 32:
            k = collections.Counter(i[0] for i in d["items"])
            q = r * M
            kind = "circle" if k.get("c", 0) >= 4 and abs(r.width - r.height) < 1 else "rect" if k.get("re", 0) == 1 or k.get("qu", 0) == 1 else "poly" if k.get("l", 0) >= 4 else "other"
            if kind != "other":
                shapes.append((q, kind))
    out = []
    for w in pg.get_text("words"):
        t = w[4].strip()
        if not 1 <= len(t) <= 4 or not re.fullmatch(r"[A-Za-z0-9]+", t):
            continue
        q = fitz.Rect(w[:4]) * M
        cx, cy = (q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2
        hit = [s for s in shapes if abs((s[0].x0 + s[0].x1) / 2 - cx) < 4 and abs((s[0].y0 + s[0].y1) / 2 - cy) < 4 and s[0].width >= q.width - 1]
        if hit:
            s = min(hit, key=lambda s: s[0].width * s[0].height)
            out.append((t, cx, cy, (round(s[0].width), round(s[0].height), s[1])))
    return out


if __name__ == "__main__":
    for sid, pages in (("mml_capitol", (2, 3, 4)), ("mml_baxter", (2, 3)), ("carpinteria_adu1", (7,))):
        d = T.doc(sid)
        for p in pages:
            pg = d[p]
            print("==", sid, p)
            for s in schedules(pg):
                print(s["kind"], s["title"], s["cols"], len(s["rows"]))
                for r in s["rows"]:
                    print("   ", {k: v for k, v in r.items() if k != "_box"})
            tg = shape_tags(pg)
            fam = collections.Counter(f for *_, f in tg)
            print(fam)
            for f in fam:
                print(f, sorted(t for t, _, _, g in tg if g == f))
