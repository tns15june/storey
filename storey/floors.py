"""One sheet per storey, their order, each floor aligned to a reference floor, and the floors as a video for SAM. Ported from lab/align."""
import random
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .meta import spark

WORK_PX = 2000   # every pixel value in the alignment code, and the gate, were tuned on sheets of this long side
GATE = 25        # inliers, unrelated real buildings reached 37 on one sample of 72 pairs, so this gate alone does not prove a fit
SCALE_GATE = 0.10   # share off the median, because accepted floors sit within 3% of their neighbours and a visibly wrong fit passed at 57 inliers, 0.92 against 1.13
STACK_PX = 960
NOT_MAIN = re.compile(r"demol|life.?safety|egress|exit|ceiling|partial|enlarged|block", re.I)
DISCHARGE = (r"\b(ground|street)\b", r"\b(first|1st|level 0*1|floor 0*1)\b")   # tried in this order
JOINT = {"type": "object", "additionalProperties": False, "required": ["bottom_to_top"],
         "properties": {"bottom_to_top": {"type": "array", "items": {"type": "integer"}}}}
JOINT_PROMPT = ("These are {n} sheets from one building's drawing set, numbered 1 to {n} in the order attached. Each shows one storey. "
                "List the sheet numbers from the lowest storey to the highest storey, using the floor names printed on the sheets.")


def ink(img) -> np.ndarray:
    """Paper-normalised ink map, 0..1, so tint, stains and film edges do not count as structure."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.medianBlur(cv2.resize(g, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA).astype(np.uint8), 15)
    bg = cv2.resize(bg.astype(np.float32), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR) + 1
    return (g / bg < 0.72).astype(np.float32)


def _apply(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:]


def warp_point(H, x, y) -> tuple[float, float]:
    px, py = _apply(H, [(x, y)])[0]
    return float(px), float(py)


def _masked(a, box):
    out = np.zeros_like(a)
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = a[y1:y2, x1:x2]
    return out


def _resized(img, long_side):
    k = long_side / max(img.shape[:2])   # up as well as down, as the lab rendered, so a small sheet still meets the tuned values and the gate
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def _coarse(c, scales=np.linspace(0.84, 1.2, 19), angles=(-2, -1, 0, 1, 2), f=5):
    """Plan-box crop of mov searched over scale and small rotation inside ref ink, by NCC at 1/f resolution. Returns (best score, H or None)."""
    a = cv2.GaussianBlur(cv2.resize(_masked(c["mov_ink"], c["mov_box"]), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    b = cv2.GaussianBlur(cv2.resize(_masked(c["ref_ink"], c["ref_box"]), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    x1, y1, x2, y2 = [v / f for v in c["mov_box"]]
    best, H = -2, None
    pad = 200
    B = cv2.copyMakeBorder(b, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = a.shape
    for s in scales:
        for th in angles:
            Rm = cv2.getRotationMatrix2D((cx, cy), th, s)
            r = cv2.warpAffine(a, Rm, (w, h))
            px, py = int((x2 - x1) * s / 2 + 2), int((y2 - y1) * s / 2 + 2)
            xa, xb, ya, yb = max(0, int(cx) - px), min(w, int(cx) + px), max(0, int(cy) - py), min(h, int(cy) + py)
            t = r[ya:yb, xa:xb]
            if t.shape[0] >= B.shape[0] or t.shape[1] >= B.shape[1]:
                continue
            _, v, _, loc = cv2.minMaxLoc(cv2.matchTemplate(B, t, cv2.TM_CCOEFF_NORMED))
            if v > best:
                best, H = v, np.array([[1, 0, loc[0] - pad - xa], [0, 1, loc[1] - pad - ya], [0, 0, 1.0]]) @ np.vstack([Rm, [0, 0, 1]])
    if H is None:
        return best, None
    S = np.diag([f, f, 1.0])
    return best, S @ H @ np.linalg.inv(S)


def _turned(c, k):
    """c with mov's ink and box turned k clockwise quarter turns, and the matrix that turned them."""
    h, w = c["mov_soft"].shape
    turn = (np.eye(3), np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1.0]]), np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1.0]]), np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1.0]]))[k]
    q = _apply(turn, [c["mov_box"][:2], c["mov_box"][2:]])
    box = (int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max()))
    return c | {"mov_ink": np.rot90(c["mov_ink"], -k).copy(), "mov_box": box}, turn


def _wide(c):
    """Cardinal orientations, then scale 0.5-2 and tilt up to 12 degrees, each scored by the coarse NCC. Returns H or None."""
    best, H = -2, None
    for k in range(4):
        turned, turn = _turned(c, k)
        v, Hk = _coarse(turned, np.geomspace(0.5, 2.0, 25), range(-12, 13, 3))
        if Hk is not None and v > best:
            best, H = v, Hk @ turn
    return H


def _matches(c, H0, patch, step, search, min_ncc, scale):
    """NCC block matches: ref point -> mov point. Blocks with little ink are skipped; RANSAC handles walls that moved."""
    ref = c["ref_soft"]
    mw = cv2.warpPerspective(c["mov_soft"], H0, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    if scale != 1.0:
        mw = cv2.resize(mw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ref = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = [int(v * scale) for v in c["ref_box"]]
    src, dst = [], []
    p = patch // 2
    for y in range(y1 + p, y2 - p, step):
        for x in range(x1 + p, x2 - p, step):
            t = ref[y - p:y + p, x - p:x + p]
            if not 0.03 < (t > 0.2).mean() < 0.7:
                continue
            ya, yb, xa, xb = y - p - search, y + p + search, x - p - search, x + p + search
            if ya < 0 or xa < 0 or yb > mw.shape[0] or xb > mw.shape[1]:
                continue
            _, v, _, loc = cv2.minMaxLoc(cv2.matchTemplate(mw[ya:yb, xa:xb], t, cv2.TM_CCOEFF_NORMED))
            if v < min_ncc:
                continue
            src.append((xa + loc[0] + p, ya + loc[1] + p))
            dst.append((x, y))
    if len(src) < 6:
        return None, None
    return _apply(np.linalg.inv(H0), np.array(src, float) / scale), np.array(dst, float) / scale


def _blocks(c, H):
    """Wide block match at half scale, then a tight full-scale pass. Returns (H, inliers), with 0 inliers if either pass finds no fit."""
    for patch, step, search, ncc, scale, thr in ((64, 24, 40, 0.35, 0.5, 8), (96, 40, 14, 0.45, 1.0, 3)):
        src, dst = _matches(c, H, patch, step, search, ncc, scale)
        if src is None:
            return H, 0
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=20000)
        if A is None or inl.sum() < 8:
            return H, 0
        H = np.vstack([A, [0, 0, 1]]).astype(np.float64)
    return H, int(inl.sum())


def _prepared(img, box, pad):
    """Ink and soft ink at the working size, the plan box padded by this share of the long side, and the matrix from sheet px to working px."""
    small = _resized(img, WORK_PX)
    h, w = small.shape[:2]
    sx, sy = w / img.shape[1], h / img.shape[0]
    p = pad * max(w, h)
    x1, y1, x2, y2 = max(0, int(box[0] * sx - p)), max(0, int(box[1] * sy - p)), min(w, int(box[2] * sx + p)), min(h, int(box[3] * sy + p))
    if x2 <= x1 or y2 <= y1:   # the box is a model's reading, so it can be anything
        x1, y1, x2, y2 = 0, 0, w, h
    a = ink(small)
    return a, cv2.GaussianBlur(a, (0, 0), 1.2), (x1, y1, x2, y2), np.diag([sx, sy, 1.0])


def _plain(c):
    """Coarse start then block match. A fit that clears the gate must also beat the same path on mov turned by a half, because on 7 of 10 real buildings walls cleared it upside down."""
    start = _coarse(c)[1]
    H, n = (np.eye(3), 0) if start is None else _blocks(c, start)
    if n >= GATE:
        turned, turn = _turned(c, 2)
        start = _coarse(turned)[1]
        H2, n2 = (H, n) if start is None else _blocks(c, start @ turn)
        if n2 > n:
            return H2, n2
    return H, n


def register(ref_img, mov_img, ref_box, mov_box) -> tuple[np.ndarray, int]:
    """Matrix from mov sheet px to ref sheet px and its inlier count, 0 if no fit held. Boxes are in sheet px. Below GATE the fit must not be used."""
    def padded(pad):
        ref_ink, ref_soft, ref_px, ref_s = _prepared(ref_img, ref_box, pad)
        mov_ink, mov_soft, mov_px, mov_s = _prepared(mov_img, mov_box, pad)
        return {"ref_ink": ref_ink, "ref_soft": ref_soft, "ref_box": ref_px, "ref_s": ref_s, "mov_ink": mov_ink, "mov_soft": mov_soft, "mov_box": mov_px, "mov_s": mov_s}

    c = padded(0.02)
    fit = _plain(c)
    if fit[1] < GATE:
        # 4% for a box that clips a small plan: mml_capitol rose from 22 and 13 inliers to 54 and 39. The wide search still runs, because 3 of 20 real quarter-turned sheets cleared the gate at 4% on a wrong fit.
        wider = _plain(padded(0.04))
        start = _wide(c)
        fit = max(fit, wider, fit if start is None else _blocks(c, start), key=lambda f: f[1])   # max keeps the first on a tie
    return np.linalg.inv(c["ref_s"]) @ fit[0] @ c["mov_s"], fit[1]


def _frame_size(shape):
    k = STACK_PX / max(shape[:2])
    return round(shape[1] * k) & ~1, round(shape[0] * k) & ~1   # H.264 at yuv420p needs even sides


def write_stack(frames: list[np.ndarray], path: Path) -> None:
    """An H.264 MP4 with one frame per floor, long side 960 px. The first frame sets the size."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH, so the floor stack cannot be written.")
    w, h = _frame_size(frames[0].shape)
    raw = b"".join(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA).tobytes() for f in frames)
    done = subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "4", "-i", "-",
                           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)], input=raw, capture_output=True)
    if done.returncode:
        raise RuntimeError("ffmpeg could not write the floor stack.")


def _plan_sheets(sheets):
    plans = [s for s in sheets if s.get("read_ok") and s.get("cls") == "floor_plan"]
    return [s for s in plans if s["number"].upper().startswith("A")] or plans


def select_floors(sheets: list[dict]) -> list[dict]:
    """One sheet per storey in set order, because the read's floor_plan class also covers life safety, demolition, framing and trade plans."""
    # shortcut: picks by sheet number prefix and title words, and a plan with no level label is left out, ask the model to pick if a real set defeats this.
    plans = _plan_sheets([s for s in sheets if s.get("level", "").strip()])
    storeys = {}
    for s in sorted(plans, key=lambda s: (bool(NOT_MAIN.search(s["title"])), s["index"])):
        storeys.setdefault(" ".join(re.sub(r"\bplans?\b", "", s["level"].lower()).split()), s)
    return sorted(storeys.values(), key=lambda s: s["index"])


def discharge_floor(levels: dict[int, str]) -> int | None:
    """The sheet labelled ground or street, then first. None if no label settles it, and the examiner chooses."""
    for pattern in DISCHARGE:
        hits = [i for i, label in levels.items() if re.search(pattern, label, re.I)]
        if hits:
            return hits[0] if len(hits) == 1 else None
    return None


def _joint_order(ctx, ids, sheet):
    """Sheet indices bottom to top from one call on shuffled sheets, or None if the reply is unusable."""
    n = len(ids)
    shown = random.Random(0).sample(ids, n)   # seeded, so a rerun hits the cache
    if shown == ids:
        shown.reverse()   # sets usually run bottom to top, and there an echoed list would look right
    paths = [ctx.dir / f"floor_{i}.jpg" for i in shown]
    for path, i in zip(paths, shown):
        cv2.imwrite(str(path), _resized(sheet(i), WORK_PX), [cv2.IMWRITE_JPEG_QUALITY, 92])   # what the order was measured on, and 3 to 6 times smaller than PNG
    try:
        picks = spark(JOINT_PROMPT.format(n=n), paths, meter=ctx.meter, schema=JOINT, fresh=ctx.fresh).get("bottom_to_top")
    except ValueError:
        return None
    if not isinstance(picks, list) or not all(type(p) is int for p in picks) or sorted(picks) != list(range(1, n + 1)):
        return None
    return [shown[p - 1] for p in picks]


def floors_stage(ctx) -> dict:
    """`transforms` map a floor's sheet px to the reference sheet's px, and a floor that failed the gates has none and is a blank frame in the stack.
    `inliers` holds each floor's count against the reference, or the link's count if it chained. The reference's entry is its total to its neighbours, which is what chose it.
    A stack pixel is a reference sheet pixel scaled by frame_size over that sheet's width and height."""
    read = ctx.stage("read")
    rows = read.get("sheets", [])
    floors = select_floors(rows)
    ids = [s["index"] for s in floors]
    box = {s["index"]: s["plan_box"] for s in floors}
    unnamed = [s["number"] or f"page {s['index'] + 1}" for s in _plan_sheets(rows) if not s.get("level", "").strip()]
    notes = [f"No floor name was read on {', '.join(unnamed)}, so stairs were not followed there."] if unnamed else []
    result = {"order": ids, "reference": ids[0] if ids else None, "transforms": {}, "inliers": {}, "unaligned": [], "stack": None, "frame_size": None,
              "discharge": discharge_floor({s["index"]: s["level"] for s in floors}), "not_checked": None}
    if len(ids) < 2:
        if "sheets" not in read:
            why = "the sheet read did not finish"
        elif unnamed or not all(s.get("read_ok") for s in rows):
            why = "fewer than two floor plans were read with a floor name"
        else:
            why = f"this set has {'one floor plan' if ids else 'no floor plans'}"
        return result | {"not_checked": " ".join([f"Stairs were not followed across floors, because {why}.", *notes])}
    paths = {s["index"]: ctx.dir / s["image"] for s in ctx.sheets}

    def sheet(i):
        return cv2.imread(str(paths[i]))   # one at a time, because a set can hold many 6000 px sheets

    order = _joint_order(ctx, ids, sheet)
    if order is None:
        order = ids
        notes.append("The floor order could not be read, so floors are in sheet order. Check the order before relying on stair continuity.")
    total = 2 * (len(order) - 1)
    score = dict.fromkeys(order, 0)
    links = {}   # (floor, neighbour) to (matrix from floor to neighbour, inliers), for fits that cleared the gate
    for done, (a, b) in enumerate(zip(order, order[1:]), 1):
        H, n = register(sheet(a), sheet(b), box[a], box[b])
        score[a] += n
        score[b] += n
        if n >= GATE:
            links[b, a], links[a, b] = (H, n), (np.linalg.inv(H), n)
        ctx.progress("Lining up the floors", done, total)
    reference = max(order, key=score.get)   # max keeps the first on a tie
    ref = sheet(reference)
    w, h = _frame_size(ref.shape)
    fits, side = {}, {}
    for done, i in enumerate((i for i in order if i != reference), total // 2 + 1):
        mov = sheet(i)
        fits[i], side[i] = register(ref, mov, box[reference], box[i]), max(mov.shape[:2])
        ctx.progress("Lining up the floors", done, total)

    def scale(H, i):
        return np.hypot(H[0, 0], H[1, 0]) * side[i] / max(ref.shape[:2])   # sheets rendered at different sizes are no scale change

    # shortcut: with two floors the median sits halfway, so this fires only past 1.22 or under 0.82, compare printed scales if a two-storey set shows a wrong fit.
    median = np.median([1.0] + [scale(H, i) for i, (H, n) in fits.items() if n >= GATE])

    def in_scale(H, i):
        return abs(scale(H, i) / median - 1) <= SCALE_GATE

    transforms = {reference: np.eye(3)} | {i: H for i, (H, n) in fits.items() if n >= GATE and in_scale(H, i)}
    inliers = {reference: score[reference]} | {i: n for i, (H, n) in fits.items()}
    outward = sorted(order, key=lambda i: abs(order.index(i) - order.index(reference)))
    for i in outward + outward[::-1]:   # out so a floor can chain through a floor that chained, then back for a floor whose good neighbour was further out
        if i in transforms:
            continue
        for j in [j for j in transforms if (i, j) in links]:
            H, n = transforms[j] @ links[i, j][0], links[i, j][1]
            if in_scale(H, i):
                transforms[i], inliers[i] = H, n
                break
    frames = []
    for i in order:
        if i in transforms:
            warped = cv2.warpPerspective(sheet(i), transforms[i], (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            frames.append(cv2.resize(warped, (w, h), interpolation=cv2.INTER_AREA))
        else:
            frames.append(np.full((h, w, 3), 255, np.uint8))   # a blank page drew no mask in the audit's control, and another building's sheet did
    write_stack(frames, ctx.dir / "stack.mp4")
    return result | {"order": order, "reference": reference, "transforms": {i: H.tolist() for i, H in transforms.items()}, "inliers": inliers,
                     "unaligned": [i for i in order if i not in transforms], "stack": "stack.mp4", "frame_size": [w, h], "not_checked": " ".join(notes) or None}
