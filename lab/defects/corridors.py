"""No-AI checks for what the stack tracker cannot see: can you walk along a corridor (band flood), and how wide is it (clear run across the band)."""
import sys
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/defects')
from common import *
from plant import draw_wall, erase
from seeds import *
from track import load_stack
from scipy import ndimage as ndi


def band(im, c):
    f, ax, lo, hi, at, a, b = c
    L = 110
    x1, y1, x2, y2 = (at - L, lo, at + L, hi) if ax == 'h' else (lo, at - L, hi, at + L)
    crop = im[y1:y2, x1:x2]
    if ax == 'v':
        crop = np.rot90(crop)
    return crop  # corridor runs left to right


def free(crop):
    I = ink(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) > 60
    I = cv2.morphologyEx(np.uint8(I), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # drop dust
    return ~cv2.dilate(I, np.ones((3, 3), np.uint8)).astype(bool)


def passable(crop):
    """Is there a free path from the left edge to the right edge of the band?"""
    F = free(crop)
    lab, n = ndi.label(F)
    return len((set(lab[:, 0]) & set(lab[:, -1])) - {0}) > 0


def clear_width(crop):
    """Across the band, the narrowest of the widest free run per column; ignores text by taking a low percentile of column widths."""
    F = free(crop)
    ws = []
    for x in range(F.shape[1]):
        best = run = 0
        for v in F[:, x]:
            run = run + 1 if v else 0
            best = max(best, run)
        ws.append(best)
    return np.array(ws)


rows = []
for b, cs in CORRIDORS.items():
    fr, names = load_stack(b)
    for i, c in enumerate(cs):
        f, ax, lo, hi, at, a, e = c
        im = fr[names.index(f)]
        w = draw_wall(im, at, lo, at, hi, 5) if ax == 'h' else draw_wall(im, lo, at, hi, at, 5)
        mid = (lo + hi) // 2
        n = draw_wall(im, a, mid, e, mid, 5) if ax == 'h' else draw_wall(im, mid, a, mid, e, 5)
        ws0, wsn = clear_width(band(im, c)), clear_width(band(n, c))
        rows.append(dict(b=b, i=i, clean_pass=bool(passable(band(im, c))), wall_pass=bool(passable(band(w, c))), narrow_pass=bool(passable(band(n, c))),
                         band_px=hi - lo, clean_w_p20=float(np.percentile(ws0, 20)), narrow_w_p20=float(np.percentile(wsn, 20)), clean_w_min=int(ws0.min()), narrow_w_min=int(wsn.min())))
        print(rows[-1])
json.dump(rows, open(OUT / 'corridors.json', 'w'))
print('wall recall', sum(not r['wall_pass'] for r in rows), '/', len(rows), 'clean blocked', sum(not r['clean_pass'] for r in rows), '/', len(rows))
print('narrow caught (p20 width drop > 30%)', sum(r['narrow_w_p20'] < .7 * r['clean_w_p20'] for r in rows), '/', len(rows))
