"""Freeze every truth file and the rules, then write a manifest with sha256 and write time. Refuses to run once a samtricks model call is on the ledger."""
import hashlib
import json
import math
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "lab/samtricks")
import fitz  # noqa: E402

import reg  # noqa: E402
import truth_viewports as tv  # noqa: E402
import truth_words as tw  # noqa: E402

fitz.TOOLS.mupdf_display_errors(False)
PDF = reg.ROOT / "data/reference/practice/bcoe_center_addendum3.pdf"
W, H = 3024.0, 2160.0
LEDGER = reg.ROOT / "data/lab/ledger.jsonl"

# Item 4. Sheet points on the 3024 x 2160 page. Boxes come from vector paths; each was checked by eye on a crop.
CLOUDS = {
    "bcoe1": [
        dict(id="A012-96", box=[1365, 693, 1383, 709], color="red", on="plan"), dict(id="A012-sign", box=[1203, 983, 1224, 1015], color="red", on="plan"),
        dict(id="A012-40", box=[1238, 1095, 1257, 1110], color="red", on="plan"), dict(id="A012-90a", box=[1293, 1539, 1311, 1554], color="red", on="plan"),
        dict(id="A012-90b", box=[1133, 1548, 1152, 1564], color="red", on="plan"), dict(id="A012-table12", box=[2194, 1842, 2290, 1870], color="red", on="table"),
        dict(id="A012-table6", box=[2196, 1907, 2281, 1924], color="red", on="table")],
    "bcoe2": [
        dict(id="E231-1", box=[343, 671, 402, 718], color="black", on="plan"), dict(id="E231-2", box=[1597, 689, 1653, 761], color="black", on="plan"),
        dict(id="E231-3", box=[1490, 690, 1542, 760], color="black", on="plan"), dict(id="E231-3red", box=[1494, 713, 1510, 739], color="red", on="plan"),
        dict(id="E231-4", box=[1289, 755, 1337, 796], color="black", on="plan"), dict(id="E231-5", box=[1299, 1104, 1372, 1161], color="black", on="plan"),
        dict(id="E231-6", box=[800, 1244, 868, 1313], color="black", on="plan")],
    "bcoe3": [
        dict(id="A111-hw", box=[2133, 316, 2166, 1015], color="grey", on="table"), dict(id="A111-127A", box=[1264, 662, 1314, 683], color="grey", on="table"),
        dict(id="A111-curtain-rows", box=[1184, 1012, 2387, 1053], color="grey", on="table"), dict(id="A111-curtain-detail", box=[402, 1767, 798, 2072], color="grey", on="detail"),
        dict(id="A111-145B", box=[1269, 947, 1392, 967], color="red", on="table"), dict(id="A111-146C", box=[1268, 987, 1391, 1006], color="red", on="table")],
}
# Item 7. Sheet points. Frame A has the addendum markup forms removed, frame B is the published page.
# The E231 form replaces a down arrow with an up arrow, so that change is real, not only a cloud.
VERSIONS = {
    "A111": dict(page=3, forms=["Fm1", "Fm2", "Fm3", "Fm4", "Fm5", "Fm6"], crop=[1000, 820, 1700, 1100], objects=[
        dict(id=1, name="cell 145.B size", box=[1269, 947, 1392, 967], kind="near_change", exists="AB"),
        dict(id=2, name="cell 146.C size", box=[1268, 987, 1391, 1006], kind="near_change", exists="AB"),
        dict(id=3, name="door type F elevation", box=[118, 222, 174, 351], kind="control", exists="AB"),
        dict(id=4, name="curtain rectangle", box=[544, 1848, 688, 2011], kind="control", exists="AB"),
        dict(id=5, name="cloud 145.B", box=[1262, 942, 1398, 972], kind="added", exists="B", seed="B"),
        dict(id=6, name="delta 6 upper", box=[1150, 935, 1186, 966], kind="added", exists="B", seed="B")]),
    "E231": dict(page=2, forms=["Fm1", "Fm2", "Fm3"], crop=[1350, 600, 1750, 860], objects=[
        dict(id=1, name="luminaire X", box=[1506, 715, 1525, 733], kind="near_change", exists="AB"),
        dict(id=2, name="old cloud around it", box=[1490, 690, 1542, 760], kind="near_change", exists="AB"),
        dict(id=3, name="hexagon 11", box=[1450, 658, 1472, 678], kind="control", exists="AB"),
        dict(id=4, name="red cloud", box=[1494, 713, 1510, 739], kind="added", exists="B", seed="B"),
        dict(id=5, name="old down arrow", box=[1498, 716, 1506, 736], kind="changed", exists="A", seed="A")]),
}


def grid(b):
    return [round(b[0] / W * 1000, 1), round(b[1] / H * 1000, 1), round(b[2] / W * 1000, 1), round(b[3] / H * 1000, 1)]


def door_tags():
    """Circles of the tag size that hold a digit, plus every other closed circle in the crop for the precision count."""
    out = {}
    # Baxter schedules 16 marks but the plan prints 15 tags: mark 9 has no tag on the door at (1756, 272), the mirror of mark 8.
    for name, f, pno, clip, nsched, ntags in (("capitol", "data/plans/mml_capitol/THE-CAPITOL.pdf", 3, (1650, 300, 2130, 1180), 8, 8), ("baxter", "data/plans/mml_baxter/THE-BAXTER.pdf", 2, (1555, 155, 2180, 1075), 16, 15)):
        pg = fitz.open(reg.ROOT / f)[pno]
        M = pg.rotation_matrix
        z = 150 / 72
        px = lambda x, y: [round((x - clip[0]) * z, 1), round((y - clip[1]) * z, 1)]
        spans = []
        for b in pg.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for s in ln["spans"]:
                    t = s["text"].strip()
                    if t:
                        r = fitz.Rect(s["bbox"]) * M
                        spans.append((t, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
        tags, other = [], []
        for g in pg.get_drawings():
            it = g["items"]
            r = fitz.Rect(g["rect"]) * M
            nc = sum(1 for i in it if i[0] == "c")
            if nc != 4 or len(it) != 4 or abs(r.width - r.height) > 1.5:
                continue
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if not (clip[0] <= cx <= clip[2] and clip[1] <= cy <= clip[3]):
                continue
            inside = [t for t, x, y in spans if math.hypot(x - cx, y - cy) < r.width / 2 + 1]
            row = dict(c=px(cx, cy), d=round(r.width * z, 1), text=inside, layer=g.get("layer"))
            if 13 < r.width < 14 and any(t.isdigit() for t in inside):
                tags.append(row)
            else:
                other.append(row)
        assert len(tags) == ntags, (name, len(tags), ntags)
        out[name] = dict(file=f, page=pno, clip=list(clip), dpi=150, schedule_marks=nsched, tags_printed=ntags, door_tags=tags, other_circles=other)
    return out


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == "__main__":
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    assert not [r for r in rows if r["name"] == "samtricks"], "a samtricks model call is already on the ledger, truth can no longer be frozen"
    T = reg.TRUTH
    T.mkdir(parents=True, exist_ok=True)
    assert set(tv.V) == set(reg.DEV + reg.HELD), set(tv.V) ^ set(reg.DEV + reg.HELD)
    assert set(tw.W) == set(reg.WORD_SHEETS)
    deltas = json.load(open(reg.OUT / "preview/deltas.json"))
    delta_truth = {f"bcoe{p}": v for p, v in deltas.items()}
    assert [len(delta_truth[k]) for k in ("bcoe1", "bcoe2", "bcoe3")] == [13, 6, 6]
    for k, v in CLOUDS.items():
        for c in v:
            c["grid"] = grid(c["box"])
    (T / "item1_viewports.json").write_text(json.dumps(dict(note=tv.__doc__, dev=reg.DEV, held=reg.HELD, sheets=tv.V), indent=1))
    (T / "item2_words.json").write_text(json.dumps(dict(note=tw.__doc__, sheets=reg.WORD_SHEETS, words=reg.WORDS, truth=tw.W), indent=1))
    (T / "item3_tags.json").write_text(json.dumps(door_tags(), indent=1))
    (T / "item4_clouds.json").write_text(json.dumps(dict(page_points=[W, H], clouds=CLOUDS, deltas=delta_truth), indent=1))
    (T / "item7_versions.json").write_text(json.dumps(VERSIONS, indent=1))
    files = sorted(T.glob("item*.json")) + [reg.ROOT / "lab/samtricks/rules.py", reg.ROOT / "lab/audit/truth.json", reg.ROOT / "lab/bakeoff/truth_meta.py"]
    man = dict(written=time.strftime("%Y-%m-%d %H:%M:%S %z"), ledger_rows_named_samtricks=0,
               files={str(p.relative_to(reg.ROOT)): dict(sha256=sha(p), mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))) for p in files})
    (T / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    d = json.load(open(T / "item3_tags.json"))
    for k, v in d.items():
        print(k, "tags", len(v["door_tags"]), "other circles", len(v["other_circles"]), sorted(t["text"][0] for t in v["door_tags"]))
