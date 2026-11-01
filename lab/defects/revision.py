"""Revision test: cycle 1 sheet followed by the same sheet re-scanned with a known change. Which items changed? Compared with a registered pixel diff."""
import sys
sys.path.insert(0, 'lab/defects')
from suite import *

REPL = 4


def items_for(b):
    fr, names = STACK[b]; f = SEEDS[b]['floor']
    it = {}
    for o in SEEDS[b]['objects']:
        bx = boxes_on(b, o, f)
        if bx:
            it[o] = tuple(bx)
    for i, c in enumerate(CORRIDORS.get(b, [])):
        if c[0] == f:
            _, ax, lo, hi, at, a, c2 = c
            it[f'corr{i}'] = (at - 40, lo - 4, at + 40, hi + 4) if ax == 'h' else (lo - 4, at - 40, hi + 4, at + 40)
    for i, e in enumerate(EXITS.get(b, [])):
        if e[0] == f:
            it[f'exit{i}'] = (e[1] - 12, e[2] - 22, e[3] + 12, e[4] + 22)
    return f, it


def rev_case(b, n):
    f, items = items_for(b)
    fr, names = STACK[b]; c1 = fr[names.index(f)].copy(); im = c1.copy()
    ks = [k for k in items if not k.startswith(('corr', 'exit'))]
    rng = random.Random(f'{b}{n}'); rng.shuffle(ks)
    changed = {}
    # one erased item, one moved item, one wall drawn over a corridor when the floor has one
    o = ks[0]; x1, y1, x2, y2 = items[o]; im = erase(im, x1 + 2, y1 + 2, x2 - 2, y2 - 2, seed=1); changed[o] = 'erased'
    if len(ks) > 1:
        o = ks[1]; x1, y1, x2, y2 = items[o]; d = max(int(.6 * (x2 - x1)), 12)
        im = move(im, x1, y1, x2, y2, d if x2 + d < im.shape[1] else -d, 0, seed=1); changed[o] = 'moved'
    cor = [k for k in items if k.startswith('corr')]
    if cor:
        k = cor[n % len(cor)]; c = [x for x in CORRIDORS[b] if x[0] == f][n % len(cor)]
        _, ax, lo, hi, at, a, c2 = c
        im = draw_wall(im, at, lo, at, hi, 5) if ax == 'h' else draw_wall(im, lo, at, hi, at, 5); changed[k] = 'walled'
    ex = [k for k in items if k.startswith('exit')]
    if ex:
        k = ex[n % len(ex)]; e = [x for x in EXITS[b] if x[0] == f][n % len(ex)]
        im = draw_wall(erase(im, e[1] - 4, e[2] - 20, e[3] + 4, e[4] + 20, seed=1), e[1], e[2], e[3], e[4], e[5]); changed[k] = 'exit inked'
    c2 = rescan(im, 10 + n)
    tag = f'rev_{b}_{n}'
    names2 = ['c1', 'c2']
    res = run([c1, c2], items, tag, 0, names2)
    # registered pixel diff, the no-AI baseline
    w2, A, cc, v = align(c2, c1, 3, (0.05, 0.05, .95, .95), fixed=False)
    g1, g2 = cv2.GaussianBlur(ink(cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)), (0, 0), 1), cv2.GaussianBlur(ink(cv2.cvtColor(w2, cv2.COLOR_BGR2GRAY)), (0, 0), 1)
    rows = {}
    i1, i2 = ink(cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)), ink(cv2.cvtColor(c2, cv2.COLOR_BGR2GRAY))
    for k, bx in items.items():
        x1, y1, x2, y2 = bx
        diff = float(np.abs(g1[y1:y2, x1:x2] - g2[y1:y2, x1:x2]).mean() / (g1[y1:y2, x1:x2].mean() + g2[y1:y2, x1:x2].mean() + 1e-6))
        r1, r2 = res[k]['c1'], res[k]['c2']
        d = dict(diff=round(diff, 3), truth=changed.get(k), c1=r1, c2=r2)
        if r1 and r2:
            b1, b2 = r1['box'], r2['box']
            a1 = i1[max(b1[1], 0):b1[3], max(b1[0], 0):b1[2]]; a2 = i2[max(b2[1], 0):b2[3], max(b2[0], 0):b2[2]]
            d['ink_ratio'] = round(float((a2 > 30).mean() / ((a1 > 30).mean() + 1e-6)), 3)
            bl = cv2.resize(a2, (a1.shape[1], a1.shape[0])) if a1.size and a2.size else a1
            d['ncc'] = round(float(np.corrcoef(cv2.GaussianBlur(a1, (0, 0), 1.5).ravel(), cv2.GaussianBlur(bl, (0, 0), 1.5).ravel())[0, 1]), 3)
            d['move'] = round(float(np.hypot(r2['cx'] - r1['cx'], r2['cy'] - r1['cy'])), 1)
        rows[k] = d
    if n == 0:
        grid([c1, c2], res, names2, items, OUT / 'thumbs' / f'g_{tag}.jpg')
    return tag, rows


if __name__ == '__main__':
    out = {}
    jobs = [(b, n) for b in SEEDS for n in range(3)]
    with ThreadPoolExecutor(6) as ex:
        for tag, rows in ex.map(lambda a: rev_case(*a), jobs):
            out[tag] = rows
            print(tag, {k: (v['truth'], v['diff'], v.get('ink_ratio'), v.get('ncc'), v.get('move'), (v['c2'] or {}).get('conf')) for k, v in rows.items()}, flush=True)
    (OUT / 'revision.json').write_text(json.dumps(out))
