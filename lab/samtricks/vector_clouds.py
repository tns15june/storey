"""No model. Cloud candidates from PDF vector paths: chains of short arcs. Used twice: as a labelling aid (every candidate is checked by eye)
and as the no-model baseline on vector sheets. Also rebuilds a sheet without its flattened Bluebeam markup forms for item 7."""
import re
import sys

import fitz
import numpy as np

sys.path.insert(0, "lab/samtricks")
import reg

fitz.TOOLS.mupdf_display_errors(False)
PDF = reg.ROOT / "data/reference/practice/bcoe_center_addendum3.pdf"


def arc_paths(pg):
    """Curved stroked paths with at least two bezier segments and a small bounding box: the bulges of a revision cloud."""
    out = []
    for g in pg.get_drawings():
        if g.get("type") not in ("s", "fs"):
            continue
        nc = sum(1 for it in g["items"] if it[0] == "c")
        r = g["rect"]
        if nc >= 2 and max(r.width, r.height) < 45 and nc >= 0.6 * len(g["items"]):
            col = g.get("color") or (0, 0, 0)
            out.append(dict(box=[r.x0, r.y0, r.x1, r.y1], color=tuple(round(c, 2) for c in col), n=nc))
    return out


def single_big_clouds(pg):
    """Clouds drawn as one long path with many beziers, the way Bluebeam flattens them."""
    return [[*g["rect"]] for g in pg.get_drawings() if g.get("type") in ("s", "fs") and sum(1 for it in g["items"] if it[0] == "c") >= 12 and max(g["rect"].width, g["rect"].height) > 40]


def cluster(boxes, gap):
    """Union boxes that sit within gap points of each other."""
    b = np.array(boxes, float)
    par = list(range(len(b)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i in range(len(b)):
        near = np.where((b[:, 0] - gap <= b[i, 2]) & (b[:, 2] + gap >= b[i, 0]) & (b[:, 1] - gap <= b[i, 3]) & (b[:, 3] + gap >= b[i, 1]))[0]
        for j in near:
            par[find(i)] = find(int(j))
    g = {}
    for i in range(len(b)):
        g.setdefault(find(i), []).append(i)
    return [dict(box=[float(b[v, 0].min()), float(b[v, 1].min()), float(b[v, 2].max()), float(b[v, 3].max())], n=len(v), ids=v) for v in g.values()]


def candidates(pno, min_arcs=6):
    pg = fitz.open(PDF)[pno]
    arcs = arc_paths(pg)
    cl = [c for c in cluster([a["box"] for a in arcs], 3.5) if c["n"] >= min_arcs] if arcs else []
    for c in cl:
        cols = [arcs[i]["color"] for i in c["ids"]]
        c["red"] = sum(1 for k in cols if k[0] > .8 and k[1] < .2) / len(cols) > .5
        c.pop("ids")
    return cl + [dict(box=b, n=99, red=True, single=True) for b in single_big_clouds(pg)]


def without_forms(pno, names):
    """A copy of the page with `/Name Do` removed for the named forms. Returns a one-page document."""
    src = fitz.open(PDF)
    d = fitz.open()
    d.insert_pdf(src, from_page=pno, to_page=pno)
    p = d[0]
    x = p.get_contents()[0]
    s = d.xref_stream(x)
    for n in names:
        s, k = re.subn(rb"/" + n.encode() + rb"\s+Do", b"", s)
        assert k == 1, (n, k)
    d.update_stream(x, s)
    return d


if __name__ == "__main__":
    for p in (1, 2, 3):
        c = candidates(p)
        print(p, len(c))
        for x in sorted(c, key=lambda r: (r["box"][1], r["box"][0])):
            print("  ", [round(v) for v in x["box"]], x["n"], "red" if x["red"] else "blk", "single" if x.get("single") else "")
