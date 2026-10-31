"""E2: stairs, rooms, exits, code facts and door schedules. Usage: e2_tasks.py run|repeat|table"""
import json
import re
import sys
import unicodedata

sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/bakeoff')
from common import *  # noqa
import cv2
import numpy as np

S13, S12, W = 'muse-spark-1.3', 'muse-spark-1.2', 'muse-wonder-1.0'
MODE = {'2k': 2000, '3k': 3000, '4k': 4000}
CFG = {'s13-low-2k': (S13, 'low', '2k'), 's13-med-2k': (S13, 'medium', '2k'), 's13-high-2k': (S13, 'high', '2k'), 's13-med-3k': (S13, 'medium', '3k'),
       's13-med-4k': (S13, 'medium', '4k'), 's12-med-2k': (S12, 'medium', '2k'), 'w-med-2k': (W, 'medium', '2k'),
       's13-med-t2': (S13, 'medium', 't2'), 's13-med-t3': (S13, 'medium', 't3'), 's13-med-strip': (S13, 'medium', 'strip'), 's13-med-zoom': (S13, 'medium', 'zoom')}

# ---- truth, set by reading the gridded crops in view/ (0-1000 sheet coordinates) ----
STAIRS = {
    'n161': {'stair': [(400, 138, 421, 198), (530, 272, 546, 315), (530, 548, 570, 576)], 'elev': [(532, 410, 545, 428)]},
    'c3': {'stair': [(753, 99, 783, 208), (677, 440, 702, 570), (753, 650, 783, 722)], 'elev': []},
    'hc1': {'stair': [(513, 198, 529, 270), (434, 495, 476, 527), (169, 545, 190, 623), (821, 545, 843, 622)], 'elev': [(417, 547, 445, 571)]},
    'bf0': {'stair': [(194, 147, 278, 197), (619, 156, 704, 202), (566, 448, 638, 505), (257, 443, 332, 500), (424, 585, 468, 694)], 'elev': []},
}
ROOMS_C3 = [('STAIR', '100'), ('LIVING', '101'), ('KITCHEN', '102'), ('CLOSET', '103'), ('CLOSET', '104'), ('POWDER', '105'), ('AWAY', '106')]
ROOMS_HC1 = {'1023': 'OFFICE', '1020': 'OFFICE', '1017': 'SUPER', '1015': 'MATRON', '1130': 'DINING', '1128': 'CORRIDOR', '1125': 'BEDROOM', '1070': 'BEDROOM',
             '1068': 'UTILITY', '1064': 'OFFICE', '1062': 'BEDROOM', '1059': 'BEDROOM', '1057': 'ALCOVE', '1055': 'BEDROOM', '1053': 'DINING', '1063': 'OFFICE',
             '1061': 'BEDROOM', '1060': 'BEDROOM', '1058': 'ALCOVE', '1056': 'BEDROOM', '1054': 'BEDROOM', '1052': 'BEDROOM', '1067': 'VISITOR', '1066': 'STAIR',
             '1005': 'PATIENT', '1002': 'CORRIDOR', '1135': 'LAVATORY', '1136': 'LINEN', '1126': 'UTILITY'}
AREAS_BF0 = [('KUTATO', 16), ('KUTATO', 20), ('KUT', 16), ('KUT', 20), ('KUT', 21), ('KUT', 21), ('ADM', 22), ('KONYVT', 25), ('SZEMINAR', 47), ('INTEZETI', 47),
             ('INT', 40), ('LH', 42), ('LH', 42), ('LH', 50), ('LH', 50), ('LH', 50), ('NAGY', 203), ('KONZULT', 47), ('RAKTAR', 26), ('KONYVTAR', 47), ('GALERIA', 98),
             ('OLT', 9), ('ET', 8), ('SZERT', 7), ('LAB', 11), ('OKT', 10)]
EXITS = {'n161': {'pts': [(394, 122), (524, 170), (619, 170), (306, 562), (673, 371), (525, 597), (537, 597)], 'cap': [180, 180, 180, 180, 360, 180, 180],
                  'dist': ['137']},
         'u2': {'pts': [(253, 516)], 'cap': [None], 'dist': ["27'-8"]}}
CODE = {'u1': dict(occupancy=r'\bB\b|business', construction=r'^\s*(type\s*)?II\s*-?\s*B\s*$', sprinklered='no', area=r'4,?351', occupant_load=r'^43$'),
        'n161': dict(occupancy=r'business', construction=r'II\s*\(?N\)?', sprinklered='yes', area=None, occupant_load=None),
        'a0': dict(occupancy=r'R-?3', construction=r'V-?B', sprinklered='conditional', area=r'400', occupant_load=None)}
DOORS_N79 = """1A03 1A04 1A05 1A06 1A07 1A08 1A09 1A10 1A11 1A12 1A13 1A15 1A15A 1AC1 1AC2 1AC3 1AC4 1AC5 1AC5A 1B01 1B02 1B03 1B04 1B04A 1B05 1B06 1B07 1B08 1B09
1B10 1B11 1B12 1B13 1B14 1B15 1B16 1B17 1B19 1B19A 1B21 1B23 1BC1 1BC2 1BC3 1BC4 1BC4A 1BS2 1BS2A 1C02 1C03 1C04 1C05 1C06 1C07 1C08 1C09 1C10 1C11 1C12 1C13 1C14 1C15 1C16
1C17 1C18 1C18A 1C19 1CC1 1CC3 1CS3 1D01 1D02 1D04 1D05 1D06 1D06A 1D07 1D08 1D09 1D10 1D11 1D12 1D13 1D14 1D15 1D16 1D17 1D18 1D19 1D20 1D21""".split()
SPECIAL_N79 = {'1AC1': ('PR 915', ''), '1B19': ('1220', ''), '1BC3': ('1065', ''), '1C11': ('PR 915', ''), '1BS2': ('915', '60'), '1CS3': ('915', '60')}
DOORS_C3 = {'1': ('36X80', 'LEFT', 'ENTRY'), '2': ('36X80', 'LEFT', 'PASSAGE'), '3': ('32X80', '-', '-'), '4': ('36X80', 'LEFT', 'CLOSET'), '5': ('60X80', 'BIFOLD', 'CLOSET'),
            '6': ('32X80', '-', '-'), '7': ('36X80', 'POCKET', 'PRIVACY'), '8': ('36X80', 'RIGHT', 'ENTRY')}
IFC = dict(json.load(open(s.OUT / 'ifc_rooms_l1.json')))

# ---- prompts and schemas ----
STAIR_P = '''You are an object grounding expert. This image is a building floor plan. Provide the bounding box coordinates of the objects in the image: "every staircase", "every elevator". A staircase is drawn as parallel tread lines, usually with an arrow and UP or DN. Include stairs inside the building and stairs attached to its outside. Do not report stair symbols in a legend or key plan. Ensure you do not miss any objects. For each object give kind, the printed label if there is one, and x_min, y_min, x_max, y_max in the 0-1000 range.'''
STAIR_S = obj(items=arr(obj(kind={"type": "string", "enum": ["stair", "elevator"]}, label=STR, x_min=INT, y_min=INT, x_max=INT, y_max=INT)))
ROOM_P = '''This image is a building floor plan. List every room label printed on it. For each room give name exactly as printed, number (the printed room number or id, empty if none), area (the printed area value, empty if none), and x, y, the centre of the label in the 0-1000 range. Do not skip small rooms or corridors.'''
ROOM_S = obj(items=arr(obj(name=STR, number=STR, area=STR, x=INT, y=INT)))
EXIT_P = '''This image is a life safety plan of one floor. Point to every exit marker on the plan: a symbol the legend calls an exit location, or an arrow tag at a building exit printed with an exit width and a capacity in people. Do not report the legend, exit signs, or fire alarm symbols. For each exit give x, y in the 0-1000 range, width as printed, and capacity (number of people, 0 if none printed). Also list every travel distance value printed on the plan.'''
EXIT_S = obj(exits=arr(obj(x=INT, y=INT, width=STR, capacity=INT)), travel_distances=arr(STR))
CODE_P = '''This image is one sheet of a building permit set. Read what it states about the building, copying values as printed. occupancy: occupancy classification or group. construction: construction type. sprinklered: yes if it states the building has a sprinkler system, no if it states it does not, conditional if it depends on a choice not yet made, not_stated if silent. area: gross or work area in square feet or as printed, empty if not printed. occupant_load: the total design occupant load, empty if not printed.'''
CODE_S = obj(occupancy=STR, construction=STR, sprinklered={"type": "string", "enum": ["yes", "no", "conditional", "not_stated"]}, area=STR, occupant_load=STR)
DOOR_P = '''This image contains a door schedule table. Read the table row by row, every row, without skipping any. For each row give door (the door mark or number), size (width and height as printed), swing (empty if there is no such column), hardware (hardware or set column, empty if none), fire_rating (empty if blank) and room (room name, empty if no such column).'''
DOOR_S = obj(rows=arr(obj(door=STR, size=STR, swing=STR, hardware=STR, fire_rating=STR, room=STR)))
TASKS = {'stairs': (STAIR_P, STAIR_S, ['n161', 'c3', 'hc1', 'bf0'], ['s13-low-2k', 's13-med-2k', 's13-high-2k', 's13-med-3k', 's13-med-4k', 's12-med-2k', 'w-med-2k', 's13-med-t2']),
         'rooms': (ROOM_P, ROOM_S, ['c3', 'hc1', 'bf0', 'n161'], ['s13-med-2k', 's12-med-2k', 's13-med-t2', 's13-med-t3', 's13-med-zoom']),
         'exits': (EXIT_P, EXIT_S, ['n161', 'u2'], ['s13-med-2k', 's12-med-2k', 'w-med-2k', 's13-med-t2']),
         'code': (CODE_P, CODE_S, ['u1', 'n161', 'a0'], ['s13-low-2k', 's13-med-2k', 's12-med-2k', 'w-med-2k', 's13-med-t2']),
         'doors': (DOOR_P, DOOR_S, ['n79', 'c3'], ['s13-med-2k', 'w-med-2k', 's13-med-strip'])}
TILE = {'t2': (2, 2, 4000), 't3': (3, 3, 6000)}
ZOOM = {'n161': (300, 130, 660, 690)}  # shortcut: drawn area set by hand for this one sheet, the product needs an ink bounding box


def zoom_tile(key, i, px=9000):
    im = s.full(key, px); h, w = im.shape[:2]
    x0, y0, x1, y1 = ZOOM[key]
    x0, x1, y0, y1 = x0 * w // 1000, x1 * w // 1000, y0 * h // 1000, y1 * h // 1000
    c, xo, yo = s.tiles(im[y0:y1, x0:x1], 2, 2)[i]
    return c, xo + x0, yo + y0, w, h


def door_strips(key, n=4):
    """Table rows only, split into n strips with the header row repeated. n79 table spans the left 73% of the sheet."""
    im = s.full(key, 6000)
    h, w = im.shape[:2]
    t = im[int(.095 * h):int(.975 * h), int(.06 * w):int(.73 * w)]
    head = t[:int(.075 * t.shape[0])]
    body = t[int(.075 * t.shape[0]):]
    cut = np.linspace(0, body.shape[0], n + 1, dtype=int)
    return [np.vstack([head, body[max(0, cut[i] - 12):cut[i + 1] + 12]]) for i in range(n)]


def jobs_for(task, keys, cfgs, runs=(0,)):
    p, sch, _, _ = TASKS[task]
    out = []
    for cfg in cfgs:
        model, eff, mode = CFG[cfg]
        for k in keys:
            for r in runs:
                if mode in MODE:
                    out.append(lambda k=k, r=r, cfg=cfg, model=model, eff=eff, mode=mode: call(task, k, cfg, r, p, lambda: [s.full(k, MODE[mode])], sch, model, eff))
                elif mode == 'zoom':
                    for i in range(4):
                        out.append(lambda k=k, r=r, i=i, cfg=cfg, model=model, eff=eff: call(task, f'{k}#{i}', cfg, r, p + ' The image is a crop of a larger sheet.', lambda: [zoom_tile(k, i)[0]], sch, model, eff))
                elif mode == 'strip':
                    if k != 'n79':
                        continue
                    for i in range(4):
                        out.append(lambda k=k, r=r, i=i, cfg=cfg, model=model, eff=eff: call(task, f'{k}#{i}', cfg, r, p + ' The image shows part of the table with its header row on top.', lambda: [door_strips(k)[i]], sch, model, eff))
                else:
                    c, rr, px = TILE[mode]
                    for i in range(c * rr):
                        out.append(lambda k=k, r=r, i=i, cfg=cfg, model=model, eff=eff, c=c, rr=rr, px=px: call(task, f'{k}#{i}', cfg, r, p + ' The image is a crop of a larger sheet.', lambda: [s.tiles(s.full(k, px), c, rr)[i][0]], sch, model, eff))
    return out


def merged(task, key, cfg, run):
    """Rows for one sheet with tile coordinates mapped back to 0-1000 of the sheet."""
    rs = [r for r in rows(task) if r['cfg'] == cfg and r['run'] == run and (r['key'] == key or r['key'].startswith(key + '#'))]
    mode = CFG[cfg][2]
    items, usd, sec, ninput = [], 0, 0, 0
    for r in rs:
        usd += r['usd']; sec += r['seconds']; ninput += r['input']
        o = r['out']
        got = o.get('items') or o.get('exits') or o.get('rows') or []
        if '#' in r['key'] and mode == 'zoom':
            cr, x0, y0, W_, H = zoom_tile(key, int(r['key'].split('#')[1]))
            h, w = cr.shape[:2]
            for it in got:
                it = dict(it, x=(x0 + it['x'] / 1000 * w) / W_ * 1000, y=(y0 + it['y'] / 1000 * h) / H * 1000)
                items.append(it)
        elif '#' in r['key'] and mode in TILE:
            c, rr, px = TILE[mode]
            im = s.full(key, px)
            H, W_ = im.shape[:2]
            cr, x0, y0 = s.tiles(im, c, rr)[int(r['key'].split('#')[1])]
            h, w = cr.shape[:2]
            fx = lambda v: (x0 + v / 1000 * w) / W_ * 1000
            fy = lambda v: (y0 + v / 1000 * h) / H * 1000
            for it in got:
                it = dict(it)
                for f in ('x', 'x_min', 'x_max'):
                    if f in it:
                        it[f] = fx(it[f])
                for f in ('y', 'y_min', 'y_max'):
                    if f in it:
                        it[f] = fy(it[f])
                items.append(it)
        else:
            items += got
    return dict(items=items, usd=usd, sec=sec, input=ninput, raw=rs)


def asc(t):
    return ''.join(c for c in unicodedata.normalize('NFD', (t or '').upper()) if unicodedata.category(c) != 'Mn').strip()


def box_center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def score_stairs(key, its):
    res = dict(found=0, want=0, err=[], iou=[], extra=0)
    used = set()
    for kind, tk in (('stair', 'stair'), ('elevator', 'elev')):
        pred = [(it['x_min'], it['y_min'], it['x_max'], it['y_max'], n) for n, it in enumerate(its) if it['kind'] == kind]
        for t in STAIRS[key][tk]:
            res['want'] += 1
            cands = [p for p in pred if p[4] not in used]
            if not cands:
                continue
            tc = box_center(t)
            d = lambda p: np.hypot(box_center(p)[0] - tc[0], box_center(p)[1] - tc[1])
            p = min(cands, key=d)
            e = d(p) / 10  # percent of sheet edge on the normalised grid
            if e <= 3.0:
                res['found'] += 1; res['err'].append(e); res['iou'].append(iou(p, t)); used.add(p[4])
    return res


def dedupe_rooms(its):
    keep = []
    for it in its:
        if not any(asc(it['name']) == asc(o['name']) and asc(it['number']) == asc(o['number']) and abs(it['x'] - o['x']) < 25 and abs(it['y'] - o['y']) < 25 for o in keep):
            keep.append(it)
    return keep


def score_rooms(key, its):
    its = dedupe_rooms(its)
    if key == 'c3':
        hit = sum(any(asc(i['number']) == n and asc(i['name']).startswith(lbl) for i in its) for lbl, n in ROOMS_C3)
        return dict(hit=hit, want=len(ROOMS_C3), n=len(its))
    if key == 'hc1':
        hit = sum(any(asc(i['number']) == n and lbl in asc(i['name']) for i in its) for n, lbl in ROOMS_HC1.items())
        return dict(hit=hit, want=len(ROOMS_HC1), n=len(its))
    if key == 'bf0':
        pool_ = [(asc(i['name']), re.sub(r'\D', '', i['area'] or i['number'])) for i in its]
        hit = 0
        for lbl, a in AREAS_BF0:
            m_ = next((p for p in pool_ if p[0].startswith(lbl) and p[1] == str(a)), None)
            if m_:
                pool_.remove(m_); hit += 1
        return dict(hit=hit, want=len(AREAS_BF0), n=len(its))
    nums = {asc(i['number']).replace(' ', '') for i in its} - {''}
    return dict(hit=len(nums & set(IFC)), want=len(IFC), n=len(its), invalid=len(nums - set(IFC)))


def score_exits(key, r):
    tr = EXITS[key]
    its = r['items']
    used, errs, caps = set(), [], 0
    for p, c in zip(tr['pts'], tr['cap']):
        cands = [(np.hypot(i['x'] - p[0], i['y'] - p[1]), n) for n, i in enumerate(its) if n not in used]
        if cands and min(cands)[0] <= 40:
            d, n = min(cands); used.add(n); errs.append(d / 10)
            caps += (c is None) or its[n]['capacity'] == c
    dists = ' '.join(x for q in r['raw'] for x in q['out']['travel_distances'])
    return dict(found=len(errs), want=len(tr['pts']), caps=caps, extra=len(its) - len(used), err=errs, dist=sum(d in dists.replace(' ', '') for d in tr['dist']), ndist=len(tr['dist']))


def score_code(key, o):
    res = {}
    for f, rx in CODE[key].items():
        if rx is None:
            res[f] = None
        elif f == 'sprinklered':
            res[f] = o[f] == rx
        else:
            res[f] = bool(re.search(rx, o[f], re.I))
    return res


def score_doors(key, its):
    if key == 'c3':
        h = 0
        for d, (size, swing, hw) in DOORS_C3.items():
            r = [i for i in its if i['door'].strip() == d]
            if r:
                z = re.sub(r'[^0-9X]', '', r[0]['size'].upper().replace('"', ''))
                h += (z == size) and (asc(r[0]['swing']).replace('N/A', '-') or '-') == swing and (asc(r[0]['hardware']) or '-') == hw
        return dict(rows=len({i['door'] for i in its}), want=8, exact=h, spec=None)
    ds = {}
    for i in its:
        ds.setdefault(asc(i['door']).replace(' ', ''), i)
    hit = len(set(ds) & set(DOORS_N79))
    spec = 0
    for d, (w_, fr) in SPECIAL_N79.items():
        if d in ds:
            spec += w_.replace(' ', '') in asc(ds[d]['size']).replace(' ', '') and (fr in ds[d]['fire_rating'] if fr else ds[d]['fire_rating'].strip() in ('', '-', 'N/A'))
    return dict(rows=hit, want=len(DOORS_N79), invented=len(set(ds) - set(DOORS_N79)), exact=None, spec=spec, nspec=len(SPECIAL_N79))


def table():
    rep = {}
    for task, (_, _, keys, cfgs) in TASKS.items():
        print('==', task)
        for cfg in cfgs:
            agg = {}
            for k in keys:
                runs = sorted({r['run'] for r in rows(task) if r['cfg'] == cfg and (r['key'] == k or r['key'].startswith(k + '#'))})
                for run in runs:
                    r = merged(task, k, cfg, run)
                    if task == 'stairs':
                        q = score_stairs(k, r['items'])
                    elif task == 'rooms':
                        q = score_rooms(k, r['items'])
                    elif task == 'exits':
                        q = score_exits(k, r)
                    elif task == 'code':
                        q = score_code(k, r['raw'][0]['out']) if len(r['raw']) == 1 else None
                    else:
                        q = score_doors(k, r['items'])
                    agg.setdefault(k, []).append((q, r['usd'], r['sec'], len(r['items'])))
            line = f'{cfg:15}'
            for k, v in agg.items():
                if task == 'stairs':
                    e = [x for q, *_ in v for x in q['err']]
                    i_ = [x for q, *_ in v for x in q['iou']]
                    line += f" {k}:{'/'.join(str(q['found']) for q, *_ in v)}of{v[0][0]['want']} n={'/'.join(str(x[3]) for x in v)} e={np.median(e) if e else -1:.2f}% iou={np.median(i_) if i_ else 0:.2f};"
                elif task == 'rooms':
                    line += f" {k}:{'/'.join(str(q['hit']) for q, *_ in v)}of{v[0][0]['want']} n={'/'.join(str(q['n']) for q, *_ in v)}" + (f" bad={v[0][0].get('invalid')}" if 'invalid' in v[0][0] else '') + ';'
                elif task == 'exits':
                    e = [x for q, *_ in v for x in q['err']]
                    line += f" {k}:{'/'.join(str(q['found']) for q, *_ in v)}of{v[0][0]['want']} cap={v[0][0]['caps']} extra={v[0][0]['extra']} dist={v[0][0]['dist']} e={np.median(e) if e else -1:.2f}%;"
                elif task == 'code':
                    for q, *_ in v:
                        line += f" {k}:" + ''.join('-' if x is None else 'Y' if x else 'x' for x in (q or {}).values())
                else:
                    line += f" {k}:{v[0][0]}"
            us = sum(x[1] for v in agg.values() for x in v) / max(1, sum(len(v) for v in agg.values()))
            se = sum(x[2] for v in agg.values() for x in v) / max(1, sum(len(v) for v in agg.values()))
            print(line, f" | ${us:.4f} {se:.0f}s/sheet")
            rep[f'{task}/{cfg}'] = line
    return rep


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'table'
    if what == 'run':
        pool([j for t in (sys.argv[2:] or TASKS) for j in jobs_for(t, TASKS[t][2], TASKS[t][3])], 8)
    if what == 'only':  # only <task> <cfg> <key> [key...]
        pool(jobs_for(sys.argv[2], sys.argv[4:], [sys.argv[3]]), 8)
    if what == 'repeat':
        task, cfgs = sys.argv[2], sys.argv[3:]
        pool(jobs_for(task, TASKS[task][2], cfgs, (1, 2)))
    table()
