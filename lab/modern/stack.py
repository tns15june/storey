"""Prepare per-set floor stacks: plan panel crop from the render, vector registration between floors, stair boxes.

Run: .venv/bin/python -I lab/modern/stack.py  (no API calls). Writes data/lab/modern/stacks/<set>_<dpi>/ and prep.json.
"""
import json, subprocess, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/modern")
import cv2, numpy as np, pymupdf as fitz
import vec
ROOT = vec.ROOT
OUT = ROOT / "data/lab/modern"
CAT = {e["id"]: e for e in json.loads((ROOT / "data/plans/catalog.json").read_text())}
# ipp is inches per point when no text layer carries the scale. Clinic composite is 1:150, read by eye from the sheet.
SETS = {
    "mml_capitol": dict(walls=("A-wall-full", "A-wall-star"), rule="A-anno-htch-docs", prompt_floor="main"),
    "mml_baxter": dict(walls=("A-wall-full", "A-wall-star"), rule="A-anno-htch-docs", prompt_floor="main"),
    "nibs_clinic": dict(wall_width=0.79, ipp=150 / 72, prompt_floor=1, rect=(800, 100, 2300, 1560)),
    "wfu_efird": dict(wall_width=0.3, ipp=None, prompt_floor=1, rect=(40, 40, 760, 1080)),
    "wfu_huffman": dict(wall_width=0.24, ipp=None, prompt_floor=1, rect=(40, 40, 760, 1080)),
}


def wall_segs(segs, cfg):
    if "walls" in cfg:
        return [s for s in segs if s[5] in cfg["walls"]]
    return [s for s in segs if abs(s[4] - cfg["wall_width"]) < 0.02]


def rot_box(pg, b):
    r = fitz.Rect(b) * pg.rotation_matrix
    return [r.x0, r.y0, r.x1, r.y1]


def wall_mask(pg, walls, rect, z=1.0):
    """Walls rasterised at z px per point inside `rect` (rotated page points)."""
    h, w = int(rect[3] - rect[1]) * int(z), int(rect[2] - rect[0]) * int(z)
    m = np.zeros((int((rect[3] - rect[1]) * z), int((rect[2] - rect[0]) * z)), np.uint8)
    M = pg.rotation_matrix
    for s in walls:
        a, b = fitz.Point(s[0], s[1]) * M, fitz.Point(s[2], s[3]) * M
        cv2.line(m, (int((a.x - rect[0]) * z), int((a.y - rect[1]) * z)), (int((b.x - rect[0]) * z), int((b.y - rect[1]) * z)), 255, 2)
    return m


def register(ref, mov, max_shift=260):
    """Integer translation (dx, dy) that moves `mov` onto `ref`: FFT cross correlation of blurred wall rasters, bounded search."""
    a, b = [cv2.GaussianBlur(x.astype(np.float32), (0, 0), 5) for x in (ref, mov)]
    a, b = a - a.mean(), b - b.mean()
    c = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), a.shape)
    c = np.fft.fftshift(c)
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    win = np.full(c.shape, -np.inf)
    win[cy - max_shift:cy + max_shift, cx - max_shift:cx + max_shift] = c[cy - max_shift:cy + max_shift, cx - max_shift:cx + max_shift]
    y, x = np.unravel_index(np.argmax(win), c.shape)
    return int(x - cx), int(y - cy), float(c[y, x] / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-9))


def write_mp4(frames, path, repeat=4):
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}", "-r", "4", "-i", "-",
                          "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        for _ in range(repeat):
            p.stdin.write(f[:h2, :w2].tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return w2, h2


def prep(id, cfg):
    fl = CAT[id]["floors"]
    pgs = [vec.open_page(id, f["file"], f["page"]) for f in fl]
    data = []
    for f, pg in zip(fl, pgs):
        segs, curves = vec.segments(pg)
        rect, note, title = (None, None, None)
        if cfg.get("rule"):
            rect, note, title = vec.plan_panel(pg, segs, rule_layer=cfg["rule"])
        walls = wall_segs(segs, cfg)
        q = (rect * pg.rotation_matrix) if rect else fitz.Rect(0, 0, pg.rect.width, pg.rect.height)
        ipp = vec.inches_per_pt(note) if note else cfg.get("ipp")
        data.append(dict(level=f["level"], pg=pg, segs=segs, walls=walls, q=q, note=note, ipp=ipp, title=title))
    # same panel on every floor: the union, so no plan is cut
    u = fitz.Rect(data[0]["q"])
    for d in data[1:]:
        u |= d["q"]
    rect = list(cfg.get("rect") or (u.x0, u.y0, u.x1, u.y1))
    ref = wall_mask(data[0]["pg"], data[0]["walls"], rect)
    regs = []
    for d in data:
        dx, dy, c = register(ref, wall_mask(d["pg"], d["walls"], rect))
        d["shift"] = (dx, dy)
        regs.append(dict(level=d["level"], dx_pt=dx, dy_pt=dy, ncc=round(c, 3)))
    return data, rect, regs


def runs_in(d, rect):
    """Stair runs inside `rect`, in rotated page points, filtered by printed scale when there is one."""
    tr = (9.0, 14.0)
    if d["ipp"]:
        runs = vec.stair_runs(d["segs"], tread_range=(tr[0] / d["ipp"], tr[1] / d["ipp"]))
        runs = [r for r in runs if 18 <= r["width"] * d["ipp"] <= 130]
    else:
        runs = vec.stair_runs(d["segs"], tread_range=(4, 9))
    out = []
    R = fitz.Rect(rect)
    for r in runs:
        b = rot_box(d["pg"], r["box"])
        if R.contains(fitz.Point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)):
            out.append(dict(box=b, steps=r["steps"], tread_pt=r["tread"], width_pt=r["width"]))
    return out


if __name__ == "__main__":
    summary = {}
    for id, cfg in SETS.items():
        data, rect, regs = prep(id, cfg)
        print(id, "panel", [round(x) for x in rect], regs)
        fl = {}
        for d in data:
            rr = runs_in(d, rect)
            # stair runs in registered frame
            dx, dy = d["shift"]
            for r in rr:
                r["box_reg"] = [r["box"][0] + dx, r["box"][1] + dy, r["box"][2] + dx, r["box"][3] + dy]
            fl[str(d["level"])] = rr
            print("  ", d["level"], d["note"], d["title"], "ipp", d["ipp"], "runs", [(r["steps"], round(r["tread_pt"], 1), round(r["width_pt"], 1), [round(x) for x in r["box_reg"]]) for r in rr])
        for dpi in (150, 200):
            Z = dpi / 72
            frames = []
            for d in data:
                im = cv2.imread(str(OUT / f"render/{id}_{d['level']}_{dpi}.png"))
                dx, dy = d["shift"]
                A = np.float32([[1, 0, dx * Z - rect[0] * Z], [0, 1, dy * Z - rect[1] * Z]])
                frames.append(cv2.warpAffine(im, A, (int((rect[2] - rect[0]) * Z), int((rect[3] - rect[1]) * Z)), borderValue=(255, 255, 255)))
            sd = OUT / f"stacks/{id}_{dpi}"
            sd.mkdir(parents=True, exist_ok=True)
            for d, f in zip(data, frames):
                cv2.imwrite(str(sd / f"{d['level']}.png"), f)
            print("  frames", frames[0].shape)
        summary[id] = dict(rect=rect, regs=regs, levels=[str(d["level"]) for d in data], runs=fl, ipp=[d["ipp"] for d in data], notes=[d["note"] for d in data])
    (OUT / "prep.json").write_text(json.dumps(summary, indent=1))
