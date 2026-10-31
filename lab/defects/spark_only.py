"""Is SAM needed? Spark's whole-sheet proposals as a standalone continuity checker, on clean sheets (already paid for) and on erased sheets."""
import sys, random
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/defects')
from suite import *
import propose_prompt as pp


def found(props, box, w, h):
    """Does any stair or shaft proposal cover at least 30% of the truth box?"""
    x1, y1, x2, y2 = box
    for k in ('stairs', 'shafts'):
        for o in props[k]:
            a, b, c, d = o['x_min'] * w / 1000, o['y_min'] * h / 1000, o['x_max'] * w / 1000, o['y_max'] * h / 1000
            iw, ih = max(0, min(x2, c) - max(x1, a)), max(0, min(y2, d) - max(y1, b))
            if iw * ih >= .3 * (x2 - x1) * (y2 - y1):
                return True
    return False


raw = json.load(open(OUT / 'seeds_raw.json'))
rows = []
# clean sheets: a real object Spark misses on an interior floor would be a false gap
for b in SEEDS:
    fr, names = STACK[b]; h, w = fr[0].shape[:2]
    for o, fl in TRUTH[b].items():
        v = {}
        for f in fl:
            bx = boxes_on(b, o, f)
            if bx:
                v[f] = found(raw[b][f], bx, w, h)
        rows.append(dict(b=b, o=o, floors=v))
tot = sum(len(r['floors']) for r in rows); hit = sum(sum(r['floors'].values()) for r in rows)
print('Spark clean: real objects found', hit, '/', tot)
fa = {}
for r in rows:
    k = list(r['floors'])
    for i in range(1, len(k) - 1):
        if not r['floors'][k[i]] and any(r['floors'][x] for x in k[:i]) and any(r['floors'][x] for x in k[i + 1:]):
            fa[r['b']] = fa.get(r['b'], 0) + 1
print('Spark clean: false gaps per building', fa)
per = {}
for r in rows:
    per.setdefault(r['b'], [0, 0]); per[r['b']][0] += sum(r['floors'].values()); per[r['b']][1] += len(r['floors'])
print(per)

ids = [c for c in cases() if c['type'] == 'erase' and c['pos'] == 'interior']
random.Random(11).shuffle(ids)
pick, seen = [], {}
for c in ids:  # at most 5 per building
    if seen.get(c['b'], 0) < 5:
        pick.append(c); seen[c['b']] = seen.get(c['b'], 0) + 1


def one(c):
    b = c['b']; fr, names = STACK[b]; h, w = fr[0].shape[:2]
    bx = boxes_on(b, c['obj'], c['floor'])
    pl = erase(fr[names.index(c['floor'])], bx[0] + 2, bx[1] + 2, bx[2] - 2, bx[3] - 2, seed=1)
    r, info = m.spark(pp.P, [pl], schema=pp.schema)
    return dict(id=c['id'], b=b, obj=c['obj'], floor=c['floor'], still_reports=found(r, bx, w, h), clean_reports=found(raw[b][c['floor']], bx, w, h), usd=info['usd'], seconds=info['seconds'])


with ThreadPoolExecutor(5) as ex:
    res = list(ex.map(one, pick))
for r in res:
    print(r)
(OUT / 'spark_only.json').write_text(json.dumps(dict(clean=rows, erased=res, hit=hit, tot=tot, fa=fa)))
caught = [r for r in res if r['clean_reports'] and not r['still_reports']]
print('Spark on erased sheets: of', len(res), 'cases,', sum(r['clean_reports'] for r in res), 'were seen on clean,', len(caught), 'vanished after erasing, usd', round(sum(r['usd'] for r in res), 3))
