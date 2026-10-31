"""Turn per-floor tracks into findings. Deterministic, no model calls. Thresholds are fitted on the dev buildings only."""
import numpy as np

P = dict(conf=.30, empty=.45, jump=.5, agree=.25, weak_dist=.5)


def _s(r):
    b = r['box']; return min(b[2] - b[0], b[3] - b[1])


def findings(res, met, names, P=P):
    """Returns a list of dicts: type (GAP, EMPTY, JUMP, TOPEND), object, floor, evidence."""
    out = []
    present = {o: [n for n in names if res[o][n] and (res[o][n]['conf'] or 0) >= P['conf']] for o in res}
    med = {}
    for o, fl in present.items():
        if fl:
            med[o] = (np.median([res[o][n]['cx'] for n in fl]), np.median([res[o][n]['cy'] for n in fl]))
    # a sheet's global offset is what the other objects agree on, so loose alignment is not read as a moved stair
    off = {}
    for n in names:
        dx = [res[o][n]['cx'] - med[o][0] for o in med if n in present[o]]
        dy = [res[o][n]['cy'] - med[o][1] for o in med if n in present[o]]
        off[n] = (float(np.median(dx)), float(np.median(dy))) if len(dx) >= 3 else (0., 0.)
    for o, fl in present.items():
        if len(fl) < 2:
            continue
        idx = [names.index(n) for n in fl]
        inks = {n: met[o][n]['ink'] for n in fl}
        solid = []
        for n in fl:
            others = [inks[x] for x in fl if x != n]
            r = inks[n] / (np.median(others) + 1e-6)
            if r < P['empty']:
                out.append(dict(type='EMPTY', obj=o, floor=n, ink_ratio=round(float(r), 3), conf=res[o][n]['conf']))
            else:
                solid.append(n)
        if len(solid) < 2:
            continue
        # after a real gap SAM often resumes at low confidence: a weak mask in the right place with the right ink still closes the bracket
        s0 = float(np.median([_s(res[o][n]) for n in solid]))
        ref = np.median([inks[x] for x in solid])
        weak = []
        for n in names:
            r = res[o][n]
            if n in fl or not r:
                continue
            d = np.hypot(r['cx'] - off[n][0] - med[o][0], r['cy'] - off[n][1] - med[o][1])
            if d < P['weak_dist'] * s0 and met[o][n]['ink'] >= P['empty'] * ref:
                weak.append(n)
        idx = [names.index(n) for n in solid + weak]
        for i in range(min(idx) + 1, max(idx)):
            n = names[i]
            if n not in fl and n not in weak:
                c = res[o][n]['conf'] if res[o][n] else None
                out.append(dict(type='GAP', obj=o, floor=n, conf=c, between=[names[max(x for x in idx if x < i)], names[min(x for x in idx if x > i)]]))
        s = float(np.median([_s(res[o][n]) for n in solid]))
        for k, n in enumerate(solid):
            if k == 0 or k == len(solid) - 1:
                continue
            a, b = solid[k - 1], solid[k + 1]
            pos = {x: (res[o][x]['cx'] - off[x][0], res[o][x]['cy'] - off[x][1]) for x in (a, n, b)}
            d_ab = np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1])
            d_n = min(np.hypot(pos[n][0] - pos[a][0], pos[n][1] - pos[a][1]), np.hypot(pos[n][0] - pos[b][0], pos[n][1] - pos[b][1]))
            if d_ab < P['agree'] * s and d_n > P['jump'] * s:
                out.append(dict(type='JUMP', obj=o, floor=n, moved_px=round(float(d_n), 1), neighbours_px=round(float(d_ab), 1)))
    return out


def base_find(res, met, names, key, thr, truth_floors=None):
    """No-AI baselines on the seed box: a floor is absent if `key` falls below thr (ncc) or below thr of its median over floors (ink)."""
    out = []
    for o in met:
        v = {n: met[o][n][key] for n in names}
        if key.startswith('ink'):
            ref = np.median(list(v.values())) + 1e-6
            on = [n for n in names if v[n] >= thr * ref]
        else:
            on = [n for n in names if v[n] >= thr]
        if len(on) < 2:
            continue
        idx = [names.index(n) for n in on]
        for i in range(min(idx) + 1, max(idx)):
            if names[i] not in on:
                out.append(dict(type='GAP', obj=o, floor=names[i]))
    return out


if __name__ == '__main__':
    # self-check on a synthetic stack: three objects on four floors, object a erased on floor 2, object b moved on floor 1
    names = ['3', '2', '1', '0']
    def rec(x, y): return dict(area=1000, box=[x - 50, y - 50, x + 50, y + 50], cx=float(x), cy=float(y), conf=.9)
    res = {'a': {n: rec(100, 100) for n in names}, 'b': {n: rec(300, 100) for n in names}, 'c': {n: rec(500, 100) for n in names}, 'd': {n: rec(700, 100) for n in names}}
    res['a']['2'] = None
    res['b']['1'] = rec(420, 100)
    met = {o: {n: dict(ink=.3, ncc=.8, ink_seed=.3) for n in names} for o in res}
    f = findings(res, met, names)
    assert {(x['type'], x['obj'], x['floor']) for x in f} == {('GAP', 'a', '2'), ('JUMP', 'b', '1')}, f
    # a global sheet offset must not fire
    for o in res:
        if res[o]['1']: res[o]['1'] = rec(int(res[o]['1']['cx']) + 80, 100)
    res['b']['1'] = rec(380, 100)
    f = findings(res, met, names)
    assert {(x['type'], x['obj'], x['floor']) for x in f} == {('GAP', 'a', '2')}, f
    # resumption at low confidence still brackets a gap
    res['c']['1'] = None; res['c']['0'] = dict(rec(500, 100), conf=.1)
    for o in res: res[o]['1'] = res[o]['1'] and rec(int(res[o]['1']['cx']) - 80, 100)
    f = findings(res, met, names)
    assert ('GAP', 'c', '1') in {(x['type'], x['obj'], x['floor']) for x in f}, f
    print('rules self-check ok')


def hybrid(res, met, names, P=P):
    """Frozen after the first full results. Presence is ink in the SAM-registered seed box, at the same 0.45 ratio; SAM's JUMP finding is kept for moved stairs."""
    out = [x for x in base_find(res, met, names, 'ink_reg', P['empty'])]
    out += [x for x in findings(res, met, names, P) if x['type'] == 'JUMP']
    return out
