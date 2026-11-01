"""Cached SAM calls, preprocessing, classical baselines and scoring for the samcraft experiments."""
import gzip
import hashlib
import multiprocessing as mp
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "lab")
import metalab as m  # noqa: E402
from data import OUT, ink_of  # noqa: E402

CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


POOL = None


def _decode(text):
    return [(r.object_id, (r.bounds.left, r.bounds.top, r.bounds.right, r.bounds.bottom), r.confidence, np.packbits(x := m.raster(r)), x.shape)
            for r in m._masks(text, m.image_segmentation_format)]


def sam(prompt, im):
    """One image call, cached on the prompt and the image bytes. Returns list of (id, box, conf, full-size bool mask).
    Uses one_bit masks (100 times smaller on the wire) and decodes in a process pool, because the parser is slow pure Python."""
    key = hashlib.sha1(prompt.encode() + cv2.imencode(".png", im)[1].tobytes()).hexdigest()
    p = CACHE / f"{key}.pkl.gz"
    if p.exists():
        rows = pickle.loads(gzip.decompress(p.read_bytes()))
    else:
        t0 = time.time()
        text = m._sam(prompt, {"type": "input_image", "image_url": m.data_url(im)}, {"include_confidence": "true", "mask_encoding": "one_bit"})
        m._log("sam-3.1", 0.0025, kind="image", seconds=round(time.time() - t0, 2))
        rows = POOL.apply(_decode, (text,)) if POOL else _decode(text)
        p.write_bytes(gzip.compress(pickle.dumps(rows)))
    h, w = im.shape[:2]
    out = []
    for oid, b, c, bits, shp in rows:
        mask = np.zeros((h, w), bool)
        r = np.unpackbits(bits)[: shp[0] * shp[1]].reshape(shp).astype(bool)
        x1, y1, x2, y2 = max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h)
        mask[y1:y2, x1:x2] = r[y1 - b[1]: y2 - b[1], x1 - b[0]: x2 - b[0]]
        out.append((int(oid), b, c, mask))
    return out


def pmap(fn, items, n=10):
    global POOL
    if POOL is None:  # forked before any thread starts
        cv2.setNumThreads(0)
        POOL = mp.get_context("fork").Pool(10)
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(fn, items))


def box_tok(b, w, h):
    return f"<|box;x1={int(b[0])};y1={int(b[1])};x2={int(b[2])};y2={int(b[3])};w={w};h={h}|>"


def pt_tok(p, w, h, neg=False):
    x, y = min(max(int(p[0]), 0), w - 1), min(max(int(p[1]), 0), h - 1)
    return ("-" if neg else "") + f"<|point;x={x};y={y};w={w};h={h}|>"


def obj_prompt(objs, w, h, frame=0):
    """objs: list of dicts with optional box, pos, neg."""
    parts = []
    for i, o in enumerate(objs, 1):
        t = box_tok(o["box"], w, h) if o.get("box") is not None else ""
        t += "".join(pt_tok(p, w, h) for p in o.get("pos", []))
        t += "".join(pt_tok(p, w, h, True) for p in o.get("neg", []))
        parts.append(f"{o.get('id', i)}{t}")
    return f"<{frame}f>" + ",".join(parts)


# Preprocessing. s is the drawing scale in px per plan px, so stroke widths stay comparable across crops.
def notext(im, s=1.0):
    """Drop small ink components: text, ticks, door numbers. Long walls survive."""
    ink = ink_of(im)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    small = (st[:, cv2.CC_STAT_WIDTH] < 45 * s) & (st[:, cv2.CC_STAT_HEIGHT] < 45 * s)
    small[0] = False
    out = im.copy()
    out[small[lab]] = 255
    return out


def thick(im, s=1.0):
    r = max(int(round(2 * s)), 1)
    ink = cv2.dilate(ink_of(im).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return np.full_like(im, 255) * (1 - ink[..., None])


def inverted(im, s=1.0):
    return 255 - im


def blueprint(im, s=1.0):
    k = ink_of(im)
    out = np.zeros_like(im)
    out[...] = (120, 60, 10)
    out[k] = (255, 245, 235)
    return out


def dtmap(im, s=1.0):
    """Distance to the nearest ink as colour. A room becomes a smooth blob with a peak in the middle."""
    d = ndi.distance_transform_edt(~ink_of(notext(im, s)))
    return cv2.applyColorMap(np.clip(d / (80 * s) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def regions(im, r, s=1.0):
    """Label free space after a dilation of r px seals door gaps. Regions grow back to the ink by watershed."""
    ink = ink_of(notext(im, s)).astype(np.uint8)
    rr = max(int(round(r * s)), 1)
    seal = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1)))
    n, lab = cv2.connectedComponents(1 - seal, connectivity=4)
    idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    lab = lab[tuple(idx)]  # each sealed pixel joins its nearest room
    lab[ink.astype(bool)] = 0
    return lab, n


def fill(im, s=1.0, r=12):
    """Paint each sealed region a random flat colour. The masks then already exist; SAM only looks them up."""
    lab, n = regions(im, r, s)
    rng = np.random.default_rng(0)
    pal = rng.integers(60, 255, (n + 1, 3), dtype=np.uint8)
    out = pal[lab]
    out[ink_of(im)] = 0
    return out


def faux_photo(im, s=1.0):
    """Wood-toned floor with noise and a soft wall shadow, to look more like a photo of a physical model."""
    ink = ink_of(im).astype(np.uint8)
    rng = np.random.default_rng(0)
    h, w = ink.shape
    grain = cv2.GaussianBlur(rng.random((h, w), np.float32), (0, 0), sigmaX=20, sigmaY=1.5)
    grain = (grain - grain.min()) / (np.ptp(grain) + 1e-6)
    floor = np.dstack([90 + 40 * grain, 140 + 50 * grain, 190 + 50 * grain])
    wall = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    shadow = cv2.GaussianBlur(wall.astype(np.float32), (0, 0), 6)
    out = floor * (1 - 0.8 * shadow[..., None])
    out[wall.astype(bool)] = (40, 40, 50)
    return out.clip(0, 255).astype(np.uint8)


PRE = {"raw": lambda im, s=1.0: im, "inv": inverted, "thick": thick, "notext": notext, "tn": lambda im, s=1.0: thick(notext(im, s), s),
       "blue": blueprint, "dt": dtmap, "fill": fill, "photo": faux_photo}


def fill_holes(mask):
    return ndi.binary_fill_holes(mask)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def seed_baseline(im, seed, r=12, s=1.0):
    """No model: the sealed region under the seed point. Returns a full-size bool mask."""
    lab, _ = regions(im, r, s)
    x, y = int(seed[0]), int(seed[1])
    v = lab[y, x]
    if v == 0:  # seed on ink: use the commonest label within 15 px
        w = lab[max(y - 15, 0):y + 15, max(x - 15, 0):x + 15]
        w = w[w > 0]
        if not w.size:
            return np.zeros(lab.shape, bool)
        v = np.bincount(w).argmax()
    return lab == v


def overlay(im, mask, gt=None, color=(0, 0, 255)):
    v = im.copy()
    v[mask] = (0.5 * v[mask] + 0.5 * np.array(color)).astype(np.uint8)
    if gt is not None:
        c, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, c, -1, (0, 180, 0), 2)
    return v


def sheet(tiles, path, cols=4, w=360):
    """Contact sheet: list of (BGR, caption)."""
    rows = []
    for i in range(0, len(tiles), cols):
        row = []
        for t, cap in tiles[i:i + cols]:
            s = w / max(t.shape[:2])
            t = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            c = np.full((w + 22, w, 3), 255, np.uint8)
            c[22:22 + t.shape[0], :t.shape[1]] = t
            cv2.putText(c, str(cap)[:46], (2, 15), 0, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            row.append(c)
        while len(row) < cols:
            row.append(np.full((w + 22, w, 3), 255, np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def encode(frames, path, repeat=4, fps=10):
    """H.264 MP4 of BGR frames, each repeated."""
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        for _ in range(repeat):
            p.stdin.write(f[:h2, :w2].tobytes())
    p.stdin.close()
    p.wait()
    return w2, h2


def _decode_video(text):
    return [(r.frame.frame_index, r.object_id, (r.bounds.left, r.bounds.top, r.bounds.right, r.bounds.bottom), r.confidence, np.packbits(x := m.raster(r)), x.shape)
            for r in m._masks(text, m.video_segmentation_format)]


def video(prompt, path, frames, direction="both"):
    """One tracker call with one_bit masks, cached on prompt and file. Returns {(frame, object): (box, conf, box-size bool mask)}."""
    key = hashlib.sha1(prompt.encode() + direction.encode() + open(path, "rb").read()).hexdigest()
    p = CACHE / f"v{key}.pkl.gz"
    if p.exists():
        rows = pickle.loads(gzip.decompress(p.read_bytes()))
    else:
        m._guard()
        t0 = time.time()
        body = open(path, "rb").read()  # bytes, not a file handle: a retried upload must resend from the start
        fid = m._retry(lambda: m.client.files.create(purpose="user_data", file=("clip.mp4", body, "video/mp4"))).id
        text = m._sam(prompt, {"type": "input_video", "file_id": fid}, {"include_confidence": "true", "propagation_direction": direction, "mask_encoding": "one_bit"})
        m._log("sam-3.1", max(frames, 50) * 0.0002, kind="video", frames=frames, seconds=round(time.time() - t0, 2))
        rows = POOL.apply(_decode_video, (text,)) if POOL else _decode_video(text)
        p.write_bytes(gzip.compress(pickle.dumps(rows)))
    return {(f, int(o)): (b, c, np.unpackbits(bits)[: s[0] * s[1]].reshape(s).astype(bool)) for f, o, b, c, bits, s in rows}


def paste(b, r, shape):
    """Put a box-size mask into a full-size canvas."""
    h, w = shape[:2]
    out = np.zeros((h, w), bool)
    x1, y1, x2, y2 = max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h)
    out[y1:y2, x1:x2] = r[y1 - b[1]: y2 - b[1], x1 - b[0]: x2 - b[0]]
    return out
