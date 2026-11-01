"""Spark reads for sheets with no text layer, and for code summaries. Every call goes through metalab.spark and is cached in data/lab/declared/spark.jsonl.

Rules from Round 1: muse-spark-1.3, whole sheet 2000 px at low for title blocks and code sheets, schedules as strips with the header
repeated at medium, life safety markers as 2 by 2 tiles at medium. Each read runs twice so a flag can require 2 of 2.
Usage: read_spark.py dev|held
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "lab")
sys.path.insert(0, "lab/declared")
import cv2  # noqa: E402
import fitz  # noqa: E402
import numpy as np  # noqa: E402
import read_text as T  # noqa: E402

CACHE = T.ROOT / "data/lab/declared/spark.jsonl"
_lock = threading.Lock()
_done = {}
if CACHE.exists():
    for line in CACHE.read_text().splitlines():
        r = json.loads(line)
        _done[(r["key"], r["run"])] = r


def obj(**p):
    return {"type": "object", "properties": p, "required": list(p), "additionalProperties": False}


def arr(i):
    return {"type": "array", "items": i}


STR = {"type": "string"}
INT = {"type": "integer"}

CODE_P = """This image is one sheet of a building drawing set. Read what it states about the building, copying values exactly as printed. Fill every field.
code_edition: the building code and its edition or year, such as International Building Code, 2012 Edition. Empty if none is printed. If a code is named with no year, give the name.
occupancy: occupancy classification or group as printed. Empty if none.
occupancy_mentions: every other place on the sheet that names an occupancy type or group, copied as printed.
construction: construction type as printed. Empty if none.
sprinklered: yes if the sheet states the building has a sprinkler system, no if it states it does not, conditional if it depends on a choice not yet made, not_stated if silent.
sprinkler_mentions: every statement on the sheet about sprinklers, copied as printed. For a table of questions with Yes and No boxes, say which box carries the mark and copy any comment.
stories: number of stories or levels as printed. Empty if none.
area: building or work area as printed. Empty if none.
occupant_load: total design occupant load as printed. Empty if none."""
CODE_S = obj(code_edition=STR, occupancy=STR, occupancy_mentions=arr(STR), construction=STR, sprinklered={"type": "string", "enum": ["yes", "no", "conditional", "not_stated"]},
             sprinkler_mentions=arr(STR), stories=STR, area=STR, occupant_load=STR)
TB_P = """This image is one sheet of a building drawing set. Read its title block. sheet_number: the drawing number printed in the title block, such as A2.1, G002 or A18-1, not a count like SHEET 161 OF 316. sheet_title: the sheet title as printed in the title block, joining its lines."""
TB_S = obj(sheet_number=STR, sheet_title=STR)
TBC_P = TB_P + """ sequence: the first number of a count like SHEET 81 OF 316, empty if there is none."""
TBC_S = obj(sheet_number=STR, sheet_title=STR, sequence=STR)
INDEX_P = """This image shows part of the sheet index or drawing list of a building drawing set. List every row in order, without skipping any. sheet_number: the sheet number or drawing number column. sequence: the sequence column if there is one, else empty. title: the description or title, exactly as printed. Include rows that say NOT USED. Skip empty rows and headers."""
INDEX_S = obj(rows=arr(obj(sheet_number=STR, sequence=STR, title=STR)))
DOOR_P = """This image contains a door schedule table. Read the table row by row, every row, without skipping any. door: the door mark or number. width: the width column or the whole size as printed. fire_rating: the fire rating column, empty if blank or a dash. hardware: the hardware set column. room: the room name column, empty if there is no such column."""
DOOR_S = obj(rows=arr(obj(door=STR, width=STR, fire_rating=STR, hardware=STR, room=STR)))
HW_P = """This image is a door hardware schedule. Each set has a set number and a list of door numbers. List every set with its door numbers exactly as printed."""
HW_S = obj(sets=arr(obj(set=STR, doors=arr(STR))))
FINISH_P = """This image contains a room finish schedule table. Read the table row by row, every row, without skipping any. room_no: the room or space number. room_name: the room name, exactly as printed."""
FINISH_S = obj(rows=arr(obj(room_no=STR, room_name=STR)))
TAG_P = """This image is part of a floor plan. Door tags are small oval or pill shaped symbols holding a door number such as 317A1, drawn beside a door. List every door tag you can read, with its text exactly as printed and x, y, the centre of the tag in the 0-1000 range of this image. Do not list room labels, keyed note circles or wall type squares."""
TAG_S = obj(tags=arr(obj(tag=STR, x=INT, y=INT)))
EXIT_P = """This image is part of a life safety plan of one floor. Point to every exit marker on the plan: a symbol the legend calls an exit location, an arrow tag at a building exit printed with an exit width and a capacity in people, or a box stating a stair well exit capacity. Do not report the legend, exit signs or fire alarm symbols. For each exit give x, y in the 0-1000 range of this image, width as printed (empty if none), and capacity, the number of people (0 if none printed). Also list every travel distance value printed on the plan, and any statement of occupant load."""
EXIT_S = obj(exits=arr(obj(x=INT, y=INT, width=STR, capacity=INT)), travel_distances=arr(STR), occupant_loads=arr(STR))


OCC_P = """This image is an occupant load table from a building code sheet. For each data row give area (the floor area as printed), factor (the floor area per occupant as printed) and occupants (the design occupant load as printed)."""
OCC_S = obj(rows=arr(obj(area=STR, factor=STR, occupants=STR)))
YN_P = """This image is a table of questions with a Yes column and a No column. For each row give item (the question), mark (yes if the X is in the Yes column, no if the X is in the No column, none if neither) and comment (any comment printed on the row)."""
YN_S = obj(rows=arr(obj(item=STR, mark={"type": "string", "enum": ["yes", "no", "none"]}, comment=STR)))


LABEL_P = """This image is one sheet of a building drawing set. sheet_number: the drawing number in the title block. sheet_title: the sheet title in the title block, joining its lines. drawing_title: the title of the main drawing as printed on the sheet outside the title block, empty if none."""
LABEL_S = obj(sheet_number=STR, sheet_title=STR, drawing_title=STR)


def render(sid, p, px, clip=None):
    """BGR array of a page or of a 0-1 clip, long side px."""
    pg = T.page(sid, p)
    W, H = pg.rect.width, pg.rect.height
    c = fitz.Rect(clip[0] * W, clip[1] * H, clip[2] * W, clip[3] * H) if clip else pg.rect
    z = px / max(c.width, c.height)
    pm = pg.get_pixmap(matrix=fitz.Matrix(z, z), clip=c, alpha=False)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3].copy()
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def tiles(sid, p, cols=2, rows=2, overlap=0.08, px=2000):
    """Clips of a page in 0-1 coordinates with overlap."""
    out = []
    for i in range(cols):
        for j in range(rows):
            w, h = (1 + overlap) / cols, (1 + overlap) / rows
            x0 = i * (1 - w) / max(cols - 1, 1)
            y0 = j * (1 - h) / max(rows - 1, 1)
            out.append((x0, y0, x0 + w, y0 + h))
    return out


def strips(sid, p, table, head_frac, n, px=2000):
    """Table clip split in n strips with the header on top. table is a 0-1 clip, head_frac the header share of its height."""
    im = render(sid, p, 6000, table)
    h = im.shape[0]
    head = im[:int(head_frac * h)]
    body = im[int(head_frac * h):]
    cut = np.linspace(0, body.shape[0], n + 1, dtype=int)
    out = []
    for i in range(n):
        s = np.vstack([head, body[max(0, cut[i] - 12):cut[i + 1] + 12]])
        k = px / max(s.shape[:2])
        out.append(cv2.resize(s, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else s)
    return out


def ask(key, run, prompt, image, schema, effort, max_tokens=12000):
    """image is a function so nothing renders on a cache hit."""
    k = (key, run)
    if k in _done:
        return _done[k]
    import metalab as m
    try:
        out, info = m.spark(prompt, images=[image()], model="muse-spark-1.3", schema=schema, effort=effort, max_tokens=max_tokens)
    except m.BudgetExceeded:
        raise
    except Exception as e:  # a truncated or refused call must not stop the batch; the missing read shows up as Cannot tell
        print("ERR", k, str(e)[:160], flush=True)
        return None
    row = dict(key=key, run=run, effort=effort, out=out, usd=info["usd"], seconds=info["seconds"], input=info["input"], output=info["output"])
    with _lock:
        _done[k] = row
        with open(CACHE, "a") as f:
            f.write(json.dumps(row) + "\n")
    return row


def pool(jobs, workers=6):
    with ThreadPoolExecutor(workers) as ex:
        return [f.result() for f in [ex.submit(j) for j in jobs]]


def code_jobs(sheets, runs=(0, 1)):
    return [(lambda s=s, p=p, r=r: ask(f"code/{s}/{p}", r, CODE_P, lambda: render(s, p, 2000), CODE_S, "low", 6000)) for s, p in sheets for r in runs]


def usc_jobs(runs=(0, 1)):
    s = "usc_leconte"
    j = code_jobs([(s, 1)], runs)
    j += [(lambda r=r: ask("index/usc", r, INDEX_P, lambda: render(s, 0, 2000, (0.05, 0.72, 0.9, 1.0)), INDEX_S, "low", 6000)) for r in runs]
    j += [(lambda p=p, r=r: ask(f"tb/usc/{p}", r, TB_P, lambda: render(s, p, 2000), TB_S, "low", 4000)) for p in range(11, 19) for r in runs]
    j += [(lambda r=r: ask("doors/usc", r, DOOR_P, lambda: render(s, 7, 2000, (0.66, 0.62, 0.905, 0.985)), DOOR_S, "medium")) for r in runs]
    j += [(lambda r=r: ask("hw/usc", r, HW_P, lambda: render(s, 7, 2000, (0.41, 0.49, 0.625, 0.985)), HW_S, "medium", 6000)) for r in runs]
    j += [(lambda r=r: ask("finish/usc", r, FINISH_P, lambda: render(s, 9, 2000, (0.35, 0.69, 0.75, 0.985)), FINISH_S, "medium", 6000)) for r in runs]
    plan = (0.08, 0.08, 0.68, 0.78)
    for i, c in enumerate(tiles(s, 4)):
        q = (plan[0] + c[0] * (plan[2] - plan[0]), plan[1] + c[1] * (plan[3] - plan[1]), plan[0] + c[2] * (plan[2] - plan[0]), plan[1] + c[3] * (plan[3] - plan[1]))
        j += [(lambda i=i, q=q, r=r: ask(f"tags/usc/{i}", r, TAG_P, lambda: render(s, 4, 2000, q), TAG_S, "medium", 6000)) for r in runs]
    for i, c in enumerate(tiles(s, 2)):
        j += [(lambda i=i, c=c, r=r: ask(f"exits/usc/{i}", r, EXIT_P, lambda: render(s, 2, 2000, c), EXIT_S, "medium", 6000)) for r in runs]
    j += [(lambda r=r: ask("occ/usc", r, OCC_P, lambda: render(s, 1, 1600, (0.01, 0.67, 0.235, 0.78)), OCC_S, "low", 3000)) for r in runs]
    j += [(lambda r=r: ask("yn/usc", r, YN_P, lambda: render(s, 1, 1600, (0.23, 0.01, 0.45, 0.175)), YN_S, "medium", 4000)) for r in runs]
    return j


CL = {"index": "Clinic_003_Index Of Drawings.pdf", "c161": "Clinic_161_Life Safety Composite First Floor Plan.pdf", "c162": "Clinic_162_Life Safety Composite Second Floor Plan.pdf",
      "d79": "Clinic_079_Door Schedule.pdf", "d80": "Clinic_080_Door Schedule.pdf", "d81": "Clinic_081_Door Schedule.pdf",
      "f76": "Clinic_076_Finish Schedule.pdf", "f77": "Clinic_077_Finish Schedule.pdf", "f78": "Clinic_078_Finish Schedule.pdf"}


def clinic_jobs(runs=(0, 1)):
    s = "nibs_clinic"
    j = code_jobs([(s, CL["c161"]), (s, CL["c162"])], runs)
    for i, c in enumerate(((0.06, 0.02, 0.305, 0.5), (0.06, 0.5, 0.305, 0.99), (0.295, 0.02, 0.535, 0.5), (0.295, 0.5, 0.535, 0.99), (0.525, 0.02, 0.765, 0.5), (0.525, 0.5, 0.765, 0.99), (0.755, 0.02, 0.995, 0.5), (0.755, 0.5, 0.995, 0.99))):
        j += [(lambda i=i, c=c, r=r: ask(f"index/clinic/{i}", r, INDEX_P, lambda: render(s, CL["index"], 2000, c), INDEX_S, "medium", 8000)) for r in runs]
    door_clip = {"d79": (0.06, 0.005, 0.73, 0.985), "d80": (0.06, 0.005, 0.73, 0.985), "d81": (0.015, 0.03, 0.70, 0.985)}  # 081 is a smaller page with its own margins
    for k in ("d79", "d80", "d81"):
        for i in range(4):
            j += [(lambda k=k, i=i, r=r: ask(f"doors/clinic/{k}/{i}", r, DOOR_P + " The image shows part of the table with its header row on top.", lambda: strips(s, CL[k], door_clip[k], 0.145, 4)[i], DOOR_S, "medium")) for r in runs]
    for k in ("f76", "f77", "f78"):
        for i in range(4):
            j += [(lambda k=k, i=i, r=r: ask(f"finish/clinic/{k}/{i}", r, FINISH_P + " The image shows part of the table with its header row on top.", lambda: strips(s, CL[k], (0.235, 0.015, 0.685, 0.975), 0.06, 4)[i], FINISH_S, "medium")) for r in runs]
    for k in ("c161", "c162"):
        for i, c in enumerate(tiles(s, CL[k])):
            j += [(lambda k=k, i=i, c=c, r=r: ask(f"exits/clinic/{k}/{i}", r, EXIT_P, lambda: render(s, CL[k], 2000, c), EXIT_S, "medium", 6000)) for r in runs]
        j += [(lambda k=k, r=r: ask(f"label/clinic/{k}", r, LABEL_P, lambda: render(s, CL[k], 2000), LABEL_S, "low", 3000)) for r in runs]
    return j


def clinic_files():
    return sorted(f.name for f in T.CLINIC.glob("*.pdf"))


TB_CLIP = (0.78, 0.84, 1.0, 1.0)


def clinic_tb_jobs(files, run=0):
    """Title block crop of each clinic sheet. A second read runs only where the first disagrees with the index."""
    s = "nibs_clinic"
    return [(lambda f=f: ask(f"tbc/{f}", run, TBC_P, lambda: render(s, f, 1000, TB_CLIP), TBC_S, "low", 2000)) for f in files]


def load(prefix):
    """Cached rows for a key prefix, as {(key, run): out}."""
    return {k: v["out"] for k, v in _done.items() if k[0].startswith(prefix)}


if __name__ == "__main__":
    scope = sys.argv[1]
    if scope == "dev":
        jobs = usc_jobs() + code_jobs([("mml_capitol", 0), ("mml_baxter", 0), ("carpinteria_adu1", 0), ("bcoe_addendum3", 1)])
    elif scope == "held":
        jobs = clinic_jobs() + code_jobs([("mml_abbot", 0), ("carpinteria_adu2", 0)]) + clinic_tb_jobs(clinic_files())
    elif scope.startswith("recheck:"):  # second title block read for the files named in a JSON list
        jobs = clinic_tb_jobs(json.load(open(scope.split(":", 1)[1])), run=1)
    else:
        raise SystemExit("scope is dev or held")
    pool(jobs)
    usd = sum(r["usd"] for r in _done.values())
    print(len(jobs), "jobs", len(_done), "cached", round(usd, 3), "usd", "ledger", os.environ.get("LAB_NAME"))
