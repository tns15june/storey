"""Text-layer readers: title blocks, cover index, drawing labels. No model. Everything is in the rendered page frame."""
import collections
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import fitz  # noqa: E402

sys.path.insert(0, "lab/declared")
import rules as R  # noqa: E402

ROOT = Path(".")
PLANS = ROOT / "data/plans"
CLINIC = PLANS / "nibs_clinic/2012-03-27-ClinicRedactedDrawings"
PDF = {"mml_capitol": PLANS / "mml_capitol/THE-CAPITOL.pdf", "mml_baxter": PLANS / "mml_baxter/THE-BAXTER.pdf", "mml_abbot": PLANS / "mml_abbot/THE-ABBOT.pdf",
       "usc_leconte": PLANS / "usc_leconte/USC_LeConte_Bid_Drawings.pdf", "carpinteria_adu1": PLANS / "carpinteria_adu1/plan1.pdf",
       "carpinteria_adu2": PLANS / "carpinteria_adu2/plan2.pdf", "af_ces_admin": PLANS / "af_ces_admin/CES_SD_drawings_admin_bldg.pdf",
       "af_ces_shops": PLANS / "af_ces_shops/CES_sd_drawings_warehouse_bldg.pdf", "wfu_efird": PLANS / "wfu_efird/Efird.pdf",
       "wfu_huffman": PLANS / "wfu_huffman/HUFFMAN-FLOOR-PLAN-G-3.pdf", "brandeis_hall": PLANS / "brandeis_hall/level-1.pdf",
       "bcoe_addendum3": ROOT / "data/reference/practice/bcoe_center_addendum3.pdf"}
SHEET_ID = r"[A-Z]{1,3}\d{0,3}[-.]?\d{1,3}(?:[.-]\d{1,3})?[A-Z]?"
SHEET_RE = re.compile(rf"^{SHEET_ID}$")


def doc(set_id):
    return fitz.open(PDF[set_id])


def page(set_id, p):
    """One rendered page. The clinic is one PDF per sheet, keyed by file name."""
    return fitz.open(CLINIC / p)[0] if set_id == "nibs_clinic" else doc(set_id)[p]


def lines(pg):
    """Text lines in the rendered frame: t, box, size, dir (reading direction in that frame)."""
    M = pg.rotation_matrix
    L = fitz.Matrix(M.a, M.b, M.c, M.d, 0, 0)
    out = []
    for b in pg.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = " ".join(s["text"].strip() for s in ln["spans"]).strip()
            if t:
                r = fitz.Rect(ln["bbox"]) * M
                d = fitz.Point(ln["dir"]) * L
                out.append(dict(t=t, x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1, size=round(ln["spans"][0]["size"], 1), dir=(round(d.x), round(d.y))))
    return out


def title_blocks(set_id, strip=0.9, boiler=0.3):
    """Per page sheet number and title from the right strip. Lines repeated across the set are boilerplate.

    Returns one dict per page. sheet_no is empty when the strip holds no id, which sends that page to Spark.
    """
    d = doc(set_id)
    pages = [lines(p) for p in d]
    W = [p.rect.width for p in d]
    rows = [[x for x in pg if x["x0"] >= strip * W[i]] for i, pg in enumerate(pages)]
    cnt = collections.Counter(t for pg in rows for t in {x["t"] for x in pg})
    limit = 0.75 * len(rows) if len(rows) <= 8 else max(3, boiler * len(rows))  # short sets repeat real titles such as PLANS & AXONS
    out = []
    for i, pg in enumerate(rows):
        ids = [x for x in pg if SHEET_RE.match(x["t"].replace(" ", ""))]
        num = max(ids, key=lambda x: x["size"], default=None)
        rest = [x for x in pg if x is not num and cnt[x["t"]] < limit]
        size = num["size"] if num else max((x["size"] for x in pg), default=0)
        rest = [x for x in rest if x["size"] >= 0.3 * size and not SHEET_RE.match(x["t"].replace(" ", ""))]
        rest = [x for x in rest if not re.fullmatch(r"[A-Z]{2,3}|\d\d[/.]\d\d[/.]\d\d(\d\d)?|\d{4}\d*/\d+", x["t"])]  # drawn-by initials and dates
        rot = sum(abs(x["dir"][1]) for x in rest) > sum(abs(x["dir"][0]) for x in rest)
        rest.sort(key=(lambda x: (round(x["x0"] / 4), x["y0"])) if rot else (lambda x: (round(x["y0"] / 4), x["x0"])))
        box = None
        if rest or num:
            q = rest + ([num] if num else [])
            box = [min(x["x0"] for x in q), min(x["y0"] for x in q), max(x["x1"] for x in q), max(x["y1"] for x in q)]
        out.append(dict(page=i, sheet_no=num["t"].replace(" ", "") if num else "", title=" ".join(x["t"] for x in rest), box=box, chars=sum(len(x["t"]) for x in pages[i])))
    return out


def cover_index(set_id, page=0):
    """Index rows on the cover: a sheet id with a title on its right. A column needs at least 4 rows, which keeps out stray references."""
    pg = doc(set_id)[page]
    ls = lines(pg)
    rows = []
    for x in ls:
        m = re.match(rf"^({SHEET_ID})\s+(\S.*)$", x["t"])
        if m and x["size"] >= 8 and any(c.isalpha() for c in m[2]) and SHEET_RE.match(m[1]):
            rows.append(dict(sheet_no=m[1], title=m[2], x0=x["x0"], y0=x["y0"], box=[x["x0"], x["y0"], x["x1"], x["y1"]]))
        elif SHEET_RE.match(x["t"]) and x["size"] >= 8:
            near = [o for o in ls if o is not x and 0 < o["x0"] - x["x1"] < 120 and abs(o["y0"] - x["y0"]) < 3 and any(c.isalpha() for c in o["t"])]
            if near:
                o = min(near, key=lambda o: o["x0"])
                rows.append(dict(sheet_no=x["t"], title=o["t"], x0=x["x0"], y0=x["y0"], box=[x["x0"], x["y0"], o["x1"], o["y1"]]))
    known = {t["sheet_no"] for t in title_blocks(set_id)}
    col = collections.defaultdict(list)
    for r in rows:
        col[round(r["x0"] / 6)].append(r)
    # a real index column mostly names sheets in the set, which keeps out code section lists like R302.12
    good = [c for c in col.values() if len(c) >= 4 and sum(r["sheet_no"] in known for r in c) >= 0.5 * len(c)]
    return [r for c in good for r in sorted(c, key=lambda r: r["y0"])]


FLOOR_PLAN = re.compile(r"\b(PLAN|LEVEL|FLOOR)\b", re.I)


def drawing_labels(pg, strip=0.9, min_size=None):
    """View titles on the sheet outside the title block: lines that name a floor and a plan, level or floor."""
    W = pg.rect.width
    ls = [x for x in lines(pg) if x["x0"] < strip * W]
    big = [x for x in ls if FLOOR_PLAN.search(x["t"]) and R.floors(x["t"]) and len(x["t"]) < 60]
    cut = min_size or 15
    return [x for x in big if x["size"] >= cut]


if __name__ == "__main__":
    for s in sys.argv[1:] or PDF:
        print("==", s)
        for t in title_blocks(s):
            print(t["page"], repr(t["sheet_no"]), repr(t["title"]), t["chars"])
        ix = cover_index(s)
        print("index", len(ix))
        for r in ix:
            print("  ", r["sheet_no"], "|", r["title"])
