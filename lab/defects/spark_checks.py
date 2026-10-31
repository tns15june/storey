"""Spark as the second reader for what tracks cannot prove. Crops are shuffled and carry no hint of the plant. Each check has clean controls."""
import sys, random, re
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/defects')
from suite import *

ANS = {"type": "object", "additionalProperties": False, "required": ["answer", "reason"], "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}, "reason": {"type": "string"}}}
TRI = {"type": "object", "additionalProperties": False, "required": ["answer", "reason"], "properties": {"answer": {"type": "string", "enum": ["continues_and_is_drawn", "should_continue_but_not_drawn", "ends_here_correctly"]}, "reason": {"type": "string"}}}


def crop(im, box, pad, size=900):
    x1, y1, x2, y2 = box; x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0); x2, y2 = min(x2 + pad, im.shape[1]), min(y2 + pad, im.shape[0])
    c = im[y1:y2, x1:x2]; s = size / max(c.shape[:2])
    return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


P_DOOR = ("This is a crop of an architectural floor plan. At the centre of the crop is a stretch of the building's exterior wall. Look only at the centre. Is there a door or an open doorway "
          "through that wall at the centre of the crop? Answer yes if a door leaf, door swing arc, or a gap in the wall lets people pass. Answer no if the wall is continuous there.")
P_WALL = ("This is a crop of an architectural floor plan showing a corridor. Can a person walk along the corridor from one end of the crop to the other, passing through the centre of the crop? "
          "Answer no if a wall or partition crosses the full width of the corridor. Text labels, thin dimension lines and door swings do not block.")
P_NARROW = ("This is a crop of an architectural floor plan showing a corridor. Is there a wall or partition running lengthwise inside the corridor, near its centre line, that reduces the clear width to about half? "
            "Answer yes or no. Ignore thin dimension lines and centre lines.")
P_END = ("Two crops of the same place on an architectural floor plan set, on adjacent storeys. Image 1 is storey {a}, which has a stair. Image 2 is storey {b}, {rel} storey {a}. "
         "Drawings may be slightly offset. Read the stair in image 1: its flights, UP or DN notes and arrows, landings, and break lines. Decide: "
         "continues_and_is_drawn if a stair is drawn in image 2 at this place; should_continue_but_not_drawn if image 1 shows the stair going on towards that storey but image 2 draws no stair here; "
         "ends_here_correctly if image 1 shows the stair stopping at its own storey and image 2 rightly has none.")


def jobs():
    J = []
    for b, es in EXITS.items():
        fr, names = STACK[b]
        for i, e in enumerate(es):
            f, x1, y1, x2, y2, t = e; im = fr[names.index(f)]
            box = (x1 - 10, y1 - 20, x2 + 10, y2 + 20)
            J.append(dict(id=f'door_clean_{b}_{i}', kind='door', truth='yes', im=crop(im, box, 70), p=P_DOOR, s=ANS))
            pl = draw_wall(erase(im, x1 - 4, y1 - 20, x2 + 4, y2 + 20, seed=1), x1, y1, x2, y2, t)
            J.append(dict(id=f'door_inked_{b}_{i}', kind='door', truth='no', im=crop(pl, box, 70), p=P_DOOR, s=ANS))
    for b, cs in CORRIDORS.items():
        fr, names = STACK[b]
        for i, c in enumerate(cs):
            f, ax, lo, hi, at, a, e = c; im = fr[names.index(f)]
            box = (at - 110, lo - 30, at + 110, hi + 30) if ax == 'h' else (lo - 30, at - 110, hi + 30, at + 110)
            mid = (lo + hi) // 2
            w = draw_wall(im, at, lo, at, hi, 5) if ax == 'h' else draw_wall(im, lo, at, hi, at, 5)
            n = draw_wall(im, a, mid, e, mid, 5) if ax == 'h' else draw_wall(im, mid, a, mid, e, 5)
            J.append(dict(id=f'walk_clean_{b}_{i}', kind='wall', truth='yes', im=crop(im, box, 0), p=P_WALL, s=ANS))
            J.append(dict(id=f'walk_walled_{b}_{i}', kind='wall', truth='no', im=crop(w, box, 0), p=P_WALL, s=ANS))
            J.append(dict(id=f'narrow_clean_{b}_{i}', kind='narrow', truth='no', im=crop(im, box, 0), p=P_NARROW, s=ANS))
            J.append(dict(id=f'narrow_narrowed_{b}_{i}', kind='narrow', truth='yes', im=crop(n, box, 0), p=P_NARROW, s=ANS))
    for b in SEEDS:
        fr, names = STACK[b]
        for o, fl in TRUTH[b].items():
            if SEEDS[b]['objects'][o][0] != 'stair':
                continue
            for end, inner in ((fl[0], fl[1]), (fl[-1], fl[-2])):
                # erased terminal floor: the stair's last storey goes missing
                bi, be = boxes_on(b, o, inner), boxes_on(b, o, end)
                if bi and be:
                    pl = erase(fr[names.index(end)], be[0] + 2, be[1] + 2, be[2] - 2, be[3] - 2, seed=1)
                    rel = 'above' if names.index(end) < names.index(inner) else 'below'
                    J.append(dict(id=f'end_erased_{b}_{o}_{end}', kind='end', truth='should_continue_but_not_drawn', im=crop(fr[names.index(inner)], bi, 40), im2=crop(pl, bi, 40), p=P_END.format(a=inner, b=end, rel=rel), s=TRI))
            # natural ends: the storey just beyond the last real one
            for k, step in ((names.index(fl[0]), -1), (names.index(fl[-1]), 1)):
                nx = k + step
                if 0 <= nx < len(names):
                    bi = boxes_on(b, o, names[k])
                    if bi:
                        rel = 'above' if step < 0 else 'below'
                        J.append(dict(id=f'end_natural_{b}_{o}_{names[nx]}', kind='end', truth='ends_here_correctly', im=crop(fr[k], bi, 40), im2=crop(fr[nx], bi, 40), p=P_END.format(a=names[k], b=names[nx], rel=rel), s=TRI))
    return J


P_LABEL = "This is one sheet of a building drawing set. Read the floor or level this sheet shows, exactly as printed in its title. Answer with that printed text only, or NONE if no floor title is printed."


def label_jobs():
    J = []
    for b in SEEDS:
        for f in CAT[b]['floors']:
            im = load_sheet(b, f, 2400)
            J.append(dict(id=f'label_{b}_{f["level"]}', kind='label', truth=str(f['level']), im=im, p=P_LABEL, s=None))
    return J


def one(j):
    imgs = [j['im']] + ([j['im2']] if 'im2' in j else [])
    r, info = m.spark(j['p'], imgs, schema=j['s'], effort='low')
    return j, r, info


if __name__ == '__main__':
    J = jobs() + label_jobs()
    random.Random(7).shuffle(J)
    print(len(J), flush=True)
    rows = []
    with ThreadPoolExecutor(5) as ex:
        for j, r, info in ex.map(one, J):
            ans = r['answer'] if isinstance(r, dict) else r
            rows.append(dict(id=j['id'], kind=j['kind'], truth=j['truth'], answer=ans, reason=r.get('reason', '')[:200] if isinstance(r, dict) else '', usd=info['usd']))
    (OUT / 'spark_checks.json').write_text(json.dumps(rows, indent=0))
    for k in ('door', 'wall', 'narrow', 'end'):
        s = [x for x in rows if x['kind'] == k]
        print(k, 'correct', sum(x['answer'] == x['truth'] for x in s), '/', len(s))
        for t in sorted({x['truth'] for x in s}):
            ss = [x for x in s if x['truth'] == t]
            from collections import Counter
            print('   truth', t, Counter(x['answer'] for x in ss))
    for x in rows:
        if x['kind'] == 'label':
            print(x['id'], '->', x['answer'])
    print('usd', round(sum(x['usd'] for x in rows), 3))
