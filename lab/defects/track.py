"""Run SAM's video tracker over a list of sheets, cache per-floor track metrics, draw crop grids."""
import sys, json, hashlib, tempfile, os
sys.path.insert(0, 'lab'); sys.path.insert(0, 'lab/defects')
import metalab as m
from common import *
REP = 4
CACHE = OUT / 'tracks'; CACHE.mkdir(exist_ok=True, parents=True)


def run(frames, seeds, tag, seed_idx, names):
    """frames: top-to-bottom BGR arrays. seeds: {name: box}, or {name: (box, floor_idx)}. Returns {obj: {floor_name: metrics or None}}."""
    p = CACHE / f'{tag}.json'
    if p.exists():
        return json.loads(p.read_text())
    mp4 = Path(tempfile.gettempdir()) / f'defects_{os.getpid()}_{abs(hash(tag))}.mp4'
    w, h, n = stack_mp4(frames, mp4, REP)
    ids = {o: i + 1 for i, o in enumerate(seeds)}
    by_frame = {}
    for o, s in seeds.items():
        box, fi = s if isinstance(s[0], (tuple, list)) else (s, seed_idx)
        by_frame.setdefault(fi * REP, []).append((ids[o], box))
    prompt = ''.join(f'<{fr}f>' + ','.join(f'{i}<|box;x1={b[0]};y1={b[1]};x2={b[2]};y2={b[3]};w={w};h={h}|>' for i, b in sorted(v)) for fr, v in sorted(by_frame.items()))
    recs = m.sam_video(prompt, mp4, n, 'both')
    mp4.unlink()
    rid = {str(v): k for k, v in ids.items()}
    res = {o: {nm: None for nm in names} for o in seeds}
    masks = {}
    first_prompt = min(by_frame)
    for r in recs:
        o = rid.get(str(r.object_id))
        if o is None:
            continue
        fi = r.frame.frame_index // REP
        # the settled frame of a sheet is the one the tracker reaches last in its direction
        settled = REP - 1 if r.frame.frame_index >= first_prompt else 0
        if r.frame.frame_index % REP != settled:
            continue
        ras = m.raster(r); b = r.bounds
        ys, xs = np.nonzero(ras)
        if len(xs) == 0:
            continue
        res[o][names[fi]] = dict(area=int(ras.sum()), box=[int(b.left), int(b.top), int(b.right), int(b.bottom)], cx=float(xs.mean() + b.left), cy=float(ys.mean() + b.top), conf=None if r.confidence is None else round(float(r.confidence), 3))
        masks[f'{o}|{names[fi]}'] = np.packbits(ras); masks[f'{o}|{names[fi]}|shape'] = np.array(ras.shape)
    p.write_text(json.dumps(res))
    if masks:
        np.savez_compressed(CACHE / f'{tag}.npz', **masks)
    return res


def grid(frames, res, names, seeds, path, pad=40, cell=230):
    """One row per object, one column per floor: crop at the seed box with the mask outline and the seed box."""
    tag = Path(path).stem
    rows = []
    for o, s in seeds.items():
        bx = s[0] if isinstance(s[0], (tuple, list)) else s
        cells = []
        for f, nm in zip(frames, names):
            x1, y1, x2, y2 = bx[0] - pad, bx[1] - pad, bx[2] + pad, bx[3] + pad
            x1, y1 = max(x1, 0), max(y1, 0)
            c = f[y1:y2, x1:x2].copy()
            cv2.rectangle(c, (pad if bx[0] >= pad else bx[0], pad if bx[1] >= pad else bx[1]), (c.shape[1] - pad, c.shape[0] - pad), (0, 200, 0), 1)
            r = res[o][nm]
            if r:
                b = r['box']
                cv2.rectangle(c, (b[0] - x1, b[1] - y1), (b[2] - x1, b[3] - y1), (0, 0, 255), 2)
            s_ = cell / max(c.shape[:2])
            c = cv2.resize(c, None, fx=s_, fy=s_)
            c = cv2.copyMakeBorder(c, 0, cell + 22 - c.shape[0], 0, cell + 6 - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            lab = f"{o[:12]} {nm} " + (f"a={r['area']} c={r['conf']}" if r else 'NONE')
            cv2.putText(c, lab, (2, cell + 15), 0, .36, (0, 0, 0) if r else (0, 0, 255), 1, cv2.LINE_AA)
            cells.append(c)
        rows.append(np.hstack(cells))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def load_stack(b):
    fl = CAT[b]['floors'][::-1]
    names = [str(f['level']) for f in fl]
    return [cv2.imread(str(OUT / 'stack' / b / f'{n}.png')) for n in names], names
