"""Is SAM confidence a usable filter for ghost masks on floors where the stair is absent? No API calls."""
import json, glob, sys
sys.path.insert(0, 'lab/seed')
from common import *
from score_sam import score
ok, gh = [], []
for p in glob.glob('data/lab/seed/sam/*__p0.json'):
    run = json.load(open(p))
    rows, false = score(run)
    bid = run['bid']
    for t in TRUTH[bid]:
        v = rows[t['id']]
        if not v['seeded']:
            continue
        for oid, f, b in run['seeds']:
            if f == v['seed_floor'] and hit(b, t['box'], .25):
                tr = run['tracks'].get(str(oid), {})
                for fl, o in v['floors'].items():
                    m = tr.get(str(fl))
                    if int(fl) != f and m and m.get('conf') is not None:
                        if o == 'ghost': gh.append((m['conf'], m['area'], t['cls'], bid, t['id'], fl))
                        if o == 'ok': ok.append((m['conf'], m['area'], t['cls'], bid, t['id'], fl))
                break
print('ok n', len(ok), np.percentile([x[0] for x in ok], [5, 25, 50, 75, 95]).round(2))
print('ghost n', len(gh), np.percentile([x[0] for x in gh], [5, 25, 50, 75, 95]).round(2))
for th in (.3, .4, .5, .6, .7, .8):
    print(th, 'ok kept', sum(x[0] >= th for x in ok), '/', len(ok), 'ghost kept', sum(x[0] >= th for x in gh), '/', len(gh))
print(sorted(gh)[-6:])
print(sorted(ok)[:8])
