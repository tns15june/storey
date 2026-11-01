"""End-to-end agent fixture: tools for the clinic, backed by saved results. Real saved data with three changes, listed in e2e_truth.json. No API."""
import json, math, sys
from pathlib import Path
sys.path.insert(0, "lab/modern"); sys.path.insert(0, "lab/agent")
import cv2, numpy as np, pymupdf as fitz
import vec  # lab/modern's vec, before routelib
import draw, routelib as RL
from build_cases import clinic_span, CLINIC55, route_img

ROOT = Path(".")
D = ROOT / "data/lab/agent"; MINE = D / "mine"
DIR = RL.K.D + "2012-03-27-ClinicRedactedDrawings/"
SHEETS = {"052": ("Clinic_052_Composite First Floor Plan.pdf", "Composite First Floor Plan, 1:150"),
          "055": ("Clinic_055_First Floor Plan Block 1C.pdf", "First Floor Plan Block 1C, 1:50"),
          "059": ("Clinic_059_Composite Second Floor Plan.pdf", "Composite Second Floor Plan, 1:150"),
          "161": ("Clinic_161_Life Safety Composite First Floor Plan.pdf", "Life Safety Composite First Floor Plan, sheet A18-1"),
          "162": ("Clinic_162_Life Safety Composite Second Floor Plan.pdf", "Life Safety Composite Second Floor Plan, sheet A18-2")}
# Seed A: treads of the south stair removed from both second floor sheets, so it truly stops at floor 1.
ERASE = {"059": (1508.5, 1304.5, 1604.5, 1346), "162": (1556, 1261, 1651, 1304)}
SEED_DIM_VALUE, SEED_DIM_MEAS = 2875, 2663  # Seed C: one agreeing row given a measured value that disagrees


def page(sheet):
    pg = fitz.open(DIR + SHEETS[sheet][0])[0]
    if sheet in ERASE:
        pg.draw_rect(fitz.Rect(ERASE[sheet]), color=(1, 1, 1), fill=(1, 1, 1), width=0)
    return pg


def dim_rows():
    rows = json.load(open(ROOT / "data/lab/modern/clinic_dims.json"))
    pg = vec.open_page("nibs_clinic", CLINIC55, 0)
    segs, _ = vec.segments(pg)
    a = vec.seg_arrays(segs)
    out = []
    for r in rows:
        if not r["measured_mm"] or r["value"] < 1000:
            continue
        meas = r["measured_mm"]
        if r["value"] == SEED_DIM_VALUE:
            meas = SEED_DIM_MEAS
        sp = clinic_span(a, r["x"], r["y"], r["vertical"])
        if sp is None:
            continue
        w, h = (9, 24) if r["vertical"] else (24, 9)
        (ax, ay), (bx, by) = sp
        m = 90
        x0, y0, x1, y1 = min(ax, bx, r["x"]) - m, min(ay, by, r["y"]) - m, max(ax, bx, r["x"]) + m, max(ay, by, r["y"]) + m
        W, H = pg.rect.width, pg.rect.height
        out.append(dict(id=f"D{len(out) + 1:02d}", printed_mm=r["value"], measured_mm=meas, status="agrees" if abs(meas - r["value"]) <= max(30, .03 * r["value"]) else "differs",
                        sheet="055", crop=[round(x0 * 1000 / W), round(y0 * 1000 / H), round(x1 * 1000 / W), round(y1 * 1000 / H)], _box=[r["x"] - w, r["y"] - h, r["x"] + w, r["y"] + h], _span=[ax, ay, bx, by]))
    return out


def rooms_index(fl):
    """Room number and name for each mask label, in the order the walls experiment wrote its rows."""
    z = np.load(RL.K.O + f"clinic_f{fl}.npz")
    lab, dg = z["lab"], z["dg"]
    rows = json.load(open(RL.K.O + f"clinic_f{fl}_rows.json"), parse_constant=lambda c: float("inf"))["rows"]
    labels = []
    for i in np.unique(lab):
        if i == 0:
            continue
        mk = lab == i
        if mk.sum() < 0.5 * RL.K.PXM ** 2:
            continue
        dt = cv2.distanceTransform(mk.astype(np.uint8), cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(dt.argmax()), dt.shape)
        if np.isfinite(dg[y, x]) and dg[y, x] > 0:
            labels.append(int(i))
    assert len(labels) == len(rows), (len(labels), len(rows))
    return {r[0]: dict(label=l, name=r[1]) for l, r in zip(labels, rows)}


class Tools:
    def __init__(self):
        self.dims = dim_rows()
        self.res = json.load(open(MINE / "routes.json"))
        self.rooms = {fl: rooms_index(fl) for fl in (1, 2)}
        self.pages = {}

    def list_sheets(self, a):
        return {"sheets": [{"sheet": k, "title": v[1]} for k, v in SHEETS.items()], "note": "Grid for view_sheet: 0-1000 of the whole sheet, x from the left, y from the top. The plans on 052 and 059 are not placed alike: a point at (x, y) on 052 is at about (x - 31, y + 97) on 059."}

    def view_sheet(self, a):
        s = str(a["sheet"])
        if s not in SHEETS:
            raise ValueError(f"unknown sheet {s}")
        pg = self.pages.setdefault(s, page(s))
        W, H = pg.rect.width, pg.rect.height
        x0, y0, x1, y1 = [float(a[k]) for k in ("x0", "y0", "x1", "y1")]
        c = (x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000)
        if c[2] - c[0] < 30 or c[3] - c[1] < 30:
            raise ValueError("crop too small, ask for at least 10 grid units each way")
        im, P, sc = draw.render(pg, c, long=1300, max_dpi=240)
        note = f"sheet {s} crop {[x0, y0, x1, y1]} at {sc * 72:.0f} dpi"
        rid = a.get("overlay_row")
        if rid and rid != "null":  # Spark sends the string "null"; that cost five tool errors in run 1
            r = next(x for x in self.dims if x["id"] == rid)
            b, sp = r["_box"], r["_span"]
            draw.box(im, P(b[0], b[1]), P(b[2], b[3]))
            draw.ticks(im, P(sp[0], sp[1]), P(sp[2], sp[3]))
            note += f"; magenta box is the printed number of row {rid}, magenta ticks are the span Storey measured"
        return note, im

    def get_stair_matrix(self, a):
        return {"floors": ["first floor", "second floor"], "note": "Saved result of one Spark call comparing the two composite floor plans, x and y on the view_sheet grid of each floor's sheet (052, 059).", "stairs": [
            {"name": "North stair", "first floor": {"present": True, "x": 404, "y": 186}, "second floor": {"present": True, "x": 385, "y": 250}},
            {"name": "Central stair", "first floor": {"present": True, "x": 534, "y": 298}, "second floor": {"present": False, "x": 0, "y": 0}},
            {"name": "South stair", "first floor": {"present": True, "x": 541, "y": 553}, "second floor": {"present": False, "x": 0, "y": 0}},
            {"name": "Service stair", "first floor": {"present": True, "x": 385, "y": 708}, "second floor": {"present": False, "x": 0, "y": 0}}]}

    def measure_route(self, a):
        fl = int(a["floor"])
        room = str(a["room"]).strip().upper()
        if room not in self.rooms.get(fl, {}):
            return f"no room {room} on floor {fl}"
        info = self.rooms[fl][room]
        r = next((x for x in self.res[str(fl)] if x["room"] == info["label"]), None)
        if r is None:
            return json.dumps(dict(room=room, name=info["name"], result="no walkable path found on Storey's wall mask"))
        pts = [tuple(q) for q in r["pts"]][::-1]
        im, P = route_img(fl, pts)
        return json.dumps(dict(room=room, name=info["name"], floor=fl, length_m=r["m"], note="Shortest route on Storey's wall mask to the nearest stair or exterior door. Overlay: green dot is the room, blue dot the end.")), im

    def read_table(self, a):
        n = a["name"]
        if n == "life_safety":
            return {"read_from": "sheets 161 and 162", "notes": ["Building is fully sprinklered", "Occupancy classification = Business", "Construction type II (N) per Uniform Building Code (U.B.C.)"],
                    "travel_distance_marked": [{"sheet": "161", "text": "137 FT./41.75 M"}, {"sheet": "162", "text": "153 FT./46.65 M"}],
                    "exits_marked": [{"sheet": "161", "exit": "Main Entry", "width_mm": 1830, "people": 360}, {"sheet": "161", "exit": "each Staff Entry door", "width_mm": 915, "people": 180}, {"sheet": "162", "exit": "each stair well exit", "people": 147}]}
        if n == "longest_routes":
            out = {}
            for fl in (1, 2):
                rows = json.load(open(RL.K.O + f"clinic_f{fl}_rows.json"), parse_constant=lambda c: float("inf"))["rows"]
                top = sorted([r for r in rows if math.isfinite(r[3]) and "ROOF" not in r[1]], key=lambda r: -r[3])[:5]
                out[f"floor {fl}"] = [dict(room=r[0], name=r[1]) for r in top]
            return {"note": "Rooms with the longest routes on Storey's wall mask. Use measure_route for each length.", **out}
        if n == "dimension_checks_055":
            return {"note": "Printed dimensions of 1000 mm or more on sheet 055 against Storey's measured span between dimension ticks. Each row has a view_sheet crop; pass overlay_row to see what was measured.",
                    "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.dims]}
        return {"tables": ["life_safety", "longest_routes", "dimension_checks_055"], "error": f"no table named {n}"}


S = lambda d: {"type": "string", "description": d}
N = lambda d: {"type": "number", "description": d}
TOOLS = [
    {"type": "function", "name": "list_sheets", "description": "List the sheets you can view.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"type": "function", "name": "view_sheet", "description": "Render a crop of a sheet as an image. Crops are on a 0-1000 grid of the whole sheet. Set overlay_row to a dimension row id such as D07 to draw what Storey measured, or leave it null.",
     "parameters": {"type": "object", "properties": {"sheet": S("Sheet number from list_sheets"), "x0": N("left"), "y0": N("top"), "x1": N("right"), "y1": N("bottom"), "overlay_row": {"type": ["string", "null"], "description": "Dimension row id, or null"}}, "required": ["sheet", "x0", "y0", "x1", "y1", "overlay_row"], "additionalProperties": False}},
    {"type": "function", "name": "get_stair_matrix", "description": "Storey's saved table of which stair is on which floor.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"type": "function", "name": "measure_route", "description": "Storey's shortest travel route from a room to the nearest exit, with an overlay image. Room numbers look like 1D23 or 2A02.",
     "parameters": {"type": "object", "properties": {"floor": {"type": "integer", "description": "1 or 2"}, "room": S("Room number")}, "required": ["floor", "room"], "additionalProperties": False}},
    {"type": "function", "name": "read_table", "description": "Read a saved table. Names: life_safety, longest_routes, dimension_checks_055.", "parameters": {"type": "object", "properties": {"name": S("Table name")}, "required": ["name"], "additionalProperties": False}},
    {"type": "function", "name": "file_finding", "description": "File one finding for the human examiner. This is the only way a finding enters the review.",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["stair_continuity", "travel_distance", "dimension_mismatch", "exit_capacity", "other_correction", "needs_examiner_judgment"]},
         "sheet": S("Sheet number the finding is on"), "location": S("Where on the sheet"), "summary": S("One sentence, the problem only"),
         "printed": S("The printed value cited, or empty"), "measured": S("The measured value cited, or empty"),
         "evidence_call_ids": {"type": "array", "items": {"type": "string"}, "description": "call_id of each tool call that backs this finding"}},
         "required": ["kind", "sheet", "location", "summary", "printed", "measured", "evidence_call_ids"], "additionalProperties": False}}]
SYSTEM = """You are the review agent in Storey, a tool that drafts a correction list for a city building department. You never approve a plan and you cannot pass one. A human examiner decides.

Review the medical and dental clinic with the tools. Compare what is declared with what is drawn. Storey's saved results can be wrong, so a tool result is evidence, not proof: before you file a finding that rests on a saved result, look at the drawing with view_sheet or at the overlay. File each real problem with file_finding, and cite the call_id of every tool call that backs it. File nothing you did not get from a tool result or see on a sheet. Do not quote code limits from memory. When you have finished, reply with one line saying so."""
USER = "Review this building. Start with list_sheets."

if __name__ == "__main__":
    t = Tools()
    print(len(t.dims), [(r["id"], r["printed_mm"], r["measured_mm"]) for r in t.dims if r["status"] == "differs"])
    print(json.dumps(t.read_table({"name": "longest_routes"}))[:500])
    print(t.measure_route({"floor": 2, "room": "2A02"})[0])
    print(t.measure_route({"floor": 1, "room": "1E27"})[0])
    for s, c in (("059", (480, 620, 560, 700)), ("162", (500, 600, 570, 670))):
        n, im = t.view_sheet(dict(sheet=s, x0=c[0], y0=c[1], x1=c[2], y1=c[3]))
        cv2.imwrite(str(MINE / f"e2e_patched_{s}.jpg"), im)
    n, im = t.view_sheet(dict(sheet="055", x0=t.dims[0]["crop"][0], y0=t.dims[0]["crop"][1], x1=t.dims[0]["crop"][2], y1=t.dims[0]["crop"][3], overlay_row="D01"))
    cv2.imwrite(str(MINE / "e2e_row.jpg"), im); print(n, im.shape)
