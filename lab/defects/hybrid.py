"""Hybrid rule, frozen before its fresh test: SAM tracks give each floor's offset, ink in the registered seed box decides presence.
Replays planting from the cached tracks, so `replay` makes no API call. `fresh` runs new jitter seeds on the frozen rule."""
import sys, random
sys.path.insert(0, 'lab/defects')
import suite, jitter
from suite import *
OUT2 = OUT / 'runs2'; OUT2.mkdir(exist_ok=True)
_base_metrics = suite.metrics


def offsets(res, names, seed_pos, exclude):
    """Per floor, the median move of the other tracked objects relative to their seed position."""
    off = {}
    for n in names:
        d = [(res[o][n]['cx'] - p[0], res[o][n]['cy'] - p[1]) for o, p in seed_pos.items() if o != exclude and res[o].get(n) and (res[o][n]['conf'] or 0) >= .3]
        off[n] = tuple(np.median(np.array(d), axis=0)) if len(d) >= 2 else (0., 0.)
    return off


def metrics2(frames, names, res, seed_boxes, seed_idx, sf):
    base = _base_metrics(frames, names, res, seed_boxes, seed_idx, sf)
    inks = [inkmap(f) for f in frames]
    h, w = frames[0].shape[:2]
    seed_pos = {o: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for o, b in seed_boxes.items() if sf.get(o, seed_idx) == seed_idx}
    for o, b in seed_boxes.items():
        off = offsets(res, names, seed_pos, o)
        ref = off[names[sf[o]]] if o in sf else (0., 0.)
        for i, n in enumerate(names):
            dx, dy = int(round(off[n][0] - ref[0])), int(round(off[n][1] - ref[1]))
            x1, y1, x2, y2 = max(b[0] + dx, 0), max(b[1] + dy, 0), min(b[2] + dx, w), min(b[3] + dy, h)
            base[o][n]['ink_reg'] = round(float((inks[i][y1:y2, x1:x2] > 30).mean()), 4) if y2 > y1 and x2 > x1 else 0.0
    return base


suite.metrics = metrics2; jitter.metrics = metrics2; suite.RES = OUT2; jitter.RES = OUT2
FORBID = m.sam_video


def replay():
    def boom(*a, **k): raise RuntimeError('replay must not call the API')
    m.sam_video = boom
    all_ = [json.load(open(p))['case'] for p in (OUT / 'runs').glob('*.json') if not p.name.endswith('_t.json')]
    jit = [c for c in all_ if c.get('jitter') is not None]
    norm = [c for c in all_ if c.get('jitter') is None]
    with ThreadPoolExecutor(10) as ex:
        list(ex.map(suite.do, norm))
        list(ex.map(jitter.jit_case, [{k: v for k, v in c.items() if k != 'jitter'} for c in jit]))
    print('replayed', len(list(OUT2.glob('*.json'))))


def fresh():
    used = {p.name[4:-5] for p in (OUT / 'runs').glob('jit_*.json')}
    cs = [c for c in cases() if c['type'] == 'erase' and c['pos'] == 'interior' and c['id'] not in used]
    random.Random(21).shuffle(cs)
    cs = [c for c in cs if c['b'] != 'schenley' or c['floor'] != SEEDS['schenley']['floor']][:16]  # the seed-floor Schenley box can jitter off the short sheet
    cl = [dict(c, id=c['id'] + 'b') for c in cases() if c['id'].endswith('_r2')]
    print(len(cs), len(cl), flush=True)
    def safe(c):
        c = dict(c, id=c['id'] + '_f2')
        try:
            return jitter.jit_case(c)
        except Exception as e:
            print('FAIL', c['id'], repr(e)[:160], flush=True)
    with ThreadPoolExecutor(6) as ex:
        print(list(ex.map(safe, cs + cl)))


if __name__ == '__main__':
    {'replay': replay, 'fresh': fresh}[sys.argv[1]]()
