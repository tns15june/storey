"""Item 7: SAM tracks chosen objects across two versions of a real sheet. Version A is the page with the addendum's flattened markup forms removed, version B is the published page.
Runs on the whole sheet and on a crop, in both frame orders. Usage: item7.py [single]"""
import json
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/samtricks")
import api as a  # noqa: E402
import cv2  # noqa: E402
import fitz  # noqa: E402
import numpy as np  # noqa: E402
import reg  # noqa: E402
import rules as R  # noqa: E402
import vector_clouds as vc  # noqa: E402

fitz.TOOLS.mupdf_display_errors(False)
T = json.load(open(reg.TRUTH / "item7_versions.json"))
OUT = reg.OUT / "item7"
OUT.mkdir(exist_ok=True)
FULL = [0, 0, 3024, 2160]


def rend(pg, clip):
    z = 2000 / (clip[2] - clip[0])
    pm = pg.get_pixmap(matrix=fitz.Matrix(z, z), clip=fitz.Rect(*clip), alpha=False)
    im = cv2.cvtColor(np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n), cv2.COLOR_RGB2BGR)
    return im[:im.shape[0] & ~1, :im.shape[1] & ~1].copy()


def frames(name, clip):
    t = T[name]
    b = rend(fitz.open(vc.PDF)[t["page"]], clip)
    aa = rend(vc.without_forms(t["page"], t["forms"])[0], clip)
    return aa, b


def clip_mp4(path, fr):
    h, w = fr[0].shape[:2]
    e = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "1", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in fr:
        e.stdin.write(f.tobytes())
    e.stdin.close()
    e.wait()


def in_view(b, clip):
    return b[0] >= clip[0] and b[1] >= clip[1] and b[2] <= clip[2] and b[3] <= clip[3]


def run(name, view, order, only=None):
    """order AB: frame 0 is version A. Returns per object, per version, box in px, conf, area. only: track one object in its own request."""
    tag = f"{name}_{view}_{order}" + (f"_obj{only}" if only else "")
    p = OUT / f"{tag}.json"
    if p.exists():
        return json.load(open(p))
    t = T[name]
    clip = FULL if view == "sheet" else t["crop"]
    aa, b = frames(name, clip)
    h, w = aa.shape[:2]
    z = w / (clip[2] - clip[0])
    px = lambda q: [int((q[0] - clip[0]) * z), int((q[1] - clip[1]) * z), int((q[2] - clip[0]) * z), int((q[3] - clip[1]) * z)]
    objs = [o for o in t["objects"] if in_view(o["box"], clip) and (only is None or o["id"] == only)]
    idx = {"A": order.index("A"), "B": order.index("B")}
    byf = {}
    for o in objs:
        seed = o.get("seed", order[0])
        byf.setdefault(idx[seed], []).append(f"{o['id']}{a_tok(px(o['box']), w, h)}")
    prompt = "".join(f"<{f}f>" + ",".join(v) for f, v in sorted(byf.items()))
    mp4 = OUT / f"{tag}.mp4"
    clip_mp4(mp4, [aa, b] if order == "AB" else [b, aa])
    recs = a.m.sam_video(prompt, mp4, frames=2, direction="both")
    masks = {}
    for r in recs:
        v = order[r.frame.frame_index]
        bb = r.bounds
        area = int(a.m.raster(r).sum())
        masks.setdefault(str(int(r.object_id)), {})[v] = dict(box=[bb.left, bb.top, bb.right, bb.bottom], conf=None if r.confidence is None else round(float(r.confidence), 3), area=area)
    o = dict(prompt=prompt, w=w, h=h, z=z, clip=clip, objs=[dict(o, px=px(o["box"])) for o in objs], masks=masks)
    p.write_text(json.dumps(o, indent=1))
    return o


def a_tok(b, w, h):
    return f"<|box;x1={b[0]};y1={b[1]};x2={b[2]};y2={b[3]};w={w};h={h}|>"


def score(o, order):
    """Frozen wording: an object present on both versions is judged on the frame it was not seeded on. An object on one version is judged on the frame where it does not exist."""
    rows = []
    for t in o["objs"]:
        m = o["masks"].get(str(t["id"]), {})
        seed = t.get("seed", order[0])
        for v in "AB":
            mk = m.get(v)
            has = bool(mk and mk["area"] > 0)
            if v not in t["exists"]:
                rows.append(dict(id=t["id"], name=t["name"], kind=t["kind"], v=v, want="gone", ok=not has, iou=None, conf=mk["conf"] if mk else None, area=mk["area"] if mk else 0))
            elif v != seed:
                iou = a.iou(mk["box"], t["px"]) if has else 0
                rows.append(dict(id=t["id"], name=t["name"], kind=t["kind"], v=v, want="hold", ok=iou >= R.IOU_T, iou=round(iou, 2), conf=mk["conf"] if mk else None))
    return rows


def diff_baseline(name):
    """No model: registered pixel diff of the two versions, connected regions on the sheet."""
    aa, b = frames(name, FULL)
    d = cv2.absdiff(aa, b).max(2) > 40
    n, _, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(d.astype(np.uint8), np.ones((9, 9), np.uint8)))
    return [[int(v) for v in st[i][:4]] for i in range(1, n)]


if __name__ == "__main__":
    jobs = [(n, v, o) for n in T for v in ("sheet", "crop") for o in ("AB", "BA")]
    res = a.pmap(lambda j: (j, run(*j)), jobs, 3)
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Post hoc: the batch run let overlapping objects compete, so each object now gets its own request, which is what the question asked.
        one = [(n, "crop", o, t["id"]) for n in T for t in T[n]["objects"] if in_view(t["box"], T[n]["crop"]) for o in ("AB", "BA")]
        one += [(n, "sheet", "AB", t["id"]) for n in T for t in T[n]["objects"]]
        tally = {}
        for j, o in a.pmap(lambda j: (j, run(*j)), one, 4):
            for r in score(o, j[2]):
                g = (j[1], r["kind"], r["want"])
                d = tally.setdefault(g, dict(ok=0, n=0, bad=[]))
                d["n"] += 1
                d["ok"] += r["ok"]
                if not r["ok"]:
                    d["bad"].append((j[0], j[2], r["name"], r["v"], r["iou"], r["conf"]))
        for g, d in sorted(tally.items()):
            print(g, f"{d['ok']}/{d['n']}", d["bad"])
        (reg.OUT / "item7_single.json").write_text(json.dumps({"/".join(k): v for k, v in tally.items()}, indent=1))
    summary = {}
    for j, o in res:
        rows = score(o, j[2])
        hold = [r for r in rows if r["want"] == "hold"]
        gone = [r for r in rows if r["want"] == "gone"]
        summary["/".join(j)] = dict(held=f"{sum(r['ok'] for r in hold)}/{len(hold)}", ghosts=f"{sum(not r['ok'] for r in gone)}/{len(gone)}", bad=[(r["name"], r["v"], r["want"], r["iou"], r["conf"]) for r in rows if not r["ok"]])
        print(j, summary["/".join(j)])
    for n in T:
        print("pixel diff regions", n, diff_baseline(n))
    (reg.OUT / "item7_score.json").write_text(json.dumps(summary, indent=1))
    rows = []
    for j, o in res:
        if j[2] != "AB":
            continue
        aa, b = frames(j[0], o["clip"])
        for t in o["objs"]:
            cv2.rectangle(aa, tuple(t["px"][:2]), tuple(t["px"][2:]), (255, 0, 255), 1)
            cv2.rectangle(b, tuple(t["px"][:2]), tuple(t["px"][2:]), (255, 0, 255), 1)
            m = o["masks"].get(str(t["id"]), {})
            for v, im in (("A", aa), ("B", b)):
                if v in m and m[v]["area"] > 0:
                    q = m[v]["box"]
                    cv2.rectangle(im, tuple(q[:2]), tuple(q[2:]), (0, 170, 0) if v in t["exists"] else (0, 0, 255), 2)
                    cv2.putText(im, f"{t['id']}:{m[v]['conf']}", (q[0], max(q[1] - 3, 12)), 0, .5, (0, 120, 0), 1, cv2.LINE_AA)
        rows.append(np.hstack([cv2.resize(aa, (1000, int(aa.shape[0] / 2))), cv2.resize(b, (1000, int(b.shape[0] / 2)))]))
    cv2.imwrite(str(reg.OUT / "item7_tracks.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("spent", round(a.m.spent("samtricks"), 3))
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        cells = []
        for n, oid, view in (("A111", 1, "crop"), ("A111", 1, "sheet"), ("E231", 1, "crop"), ("E231", 5, "crop"), ("E231", 4, "sheet")):
            o = run(n, view, "AB", oid)
            aa, b = frames(n, o["clip"])
            t = o["objs"][0]
            pad = 90
            x0, y0, x1, y1 = max(t["px"][0] - pad, 0), max(t["px"][1] - pad, 0), t["px"][2] + pad, t["px"][3] + pad
            m = o["masks"].get(str(oid), {})
            for v, im in (("A", aa), ("B", b)):
                cv2.rectangle(im, tuple(t["px"][:2]), tuple(t["px"][2:]), (255, 0, 255), 1)
                if v in m and m[v]["area"] > 0:
                    cv2.rectangle(im, tuple(m[v]["box"][:2]), tuple(m[v]["box"][2:]), (0, 170, 0), 1)
                c = cv2.resize(im[y0:y1, x0:x1], (480, 240), interpolation=cv2.INTER_CUBIC)
                cv2.putText(c, f"{n} {t['name']} {view} {v} {m.get(v, {}).get('conf')}", (4, 14), 0, .42, (0, 0, 255), 1, cv2.LINE_AA)
                cells.append(c)
        cv2.imwrite(str(reg.OUT / "item7_single.jpg"), np.vstack([np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
