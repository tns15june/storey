"""Numbers for report.md that need no API: Spark-only recall, duplicate-sheet check, union rule, Wilson intervals, spend."""
import sys, json, itertools
sys.path.insert(0, 'lab/defects'); sys.path.insert(0, 'lab')
from eval_hybrid import *
from common import load_sheet
from plant import rescan
from track import load_stack
from seeds import SEEDS, TRUTH
import cv2
from math import sqrt


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / 2 / n) / d; h = z * sqrt(p * (1 - p) / n + z * z / 4 / n / n) / d
    return f'{k}/{n} = {100 * p:.1f}% (95% CI {100 * (c - h):.0f}-{100 * (c + h):.0f})'


ER = [r for r in main if lab(r['case']) == 'erase_interior']
hit = lambda f, r: any(x['obj'] == r['case']['obj'] and x['floor'] == r['case']['floor'] for x in f(r))
union = lambda r: fn(r) + sam(r)
out = {}
for nm, f in (('SAM track rule', sam), ('ink fixed box .45', ink), ('hybrid', fn), ('hybrid or SAM rule', union)):
    h = [r for r in ER if r['case']['b'] not in DEV]; d = [r for r in ER if r['case']['b'] in DEV]
    T = [r for r in main if lab(r['case']) == 'erase_terminal']
    out[nm] = dict(all=wilson(sum(hit(f, r) for r in ER), len(ER)), dev=wilson(sum(hit(f, r) for r in d), len(d)), hold=wilson(sum(hit(f, r) for r in h), len(h)), terminal=wilson(sum(hit(f, r) for r in T), len(T)))
    print(nm, out[nm])
for nm, rs in (('jit first', old), ('jit fresh', new)):
    for n2, f in (('hybrid', fn), ('SAM rule', sam), ('ink box', ink), ('hybrid or SAM', union)):
        e = [r for r in rs if r['case']['type'] == 'erase']
        fa = sum(len([x for x in f(r) if x['obj'] != r['case'].get('obj')]) for r in rs)
        print(nm, n2, wilson(sum(hit(f, r) for r in e), len(e)), 'false', fa)
# per building false alarms on clean stacks, all rules; each clean stack ran 3 times counting the prior run
for nm, f in (('SAM rule', sam), ('hybrid', fn)):
    c = defaultdict(list)
    for r in main:
        if r['case']['type'] == 'clean':
            c[r['case']['b']].append(sorted((x['type'], x['obj'], x['floor']) for x in f(r)))
    print('clean', nm, dict(c))

# Spark-only, like for like: the erased floor must be bracketed by floors where Spark saw the object on clean sheets
S = json.load(open(OUT / 'spark_only.json'))
cl = {(r['b'], r['o']): r['floors'] for r in S['clean']}
ok = 0
for r in S['erased']:
    v = cl[(r['b'], r['obj'])]; k = list(v); i = k.index(r['floor']) if r['floor'] in k else -1
    br = any(v[x] for x in k[:i]) and any(v[x] for x in k[i + 1:])
    ok += (not r['still_reports']) and br
print('Spark-only continuity recall', wilson(ok, len(S['erased'])), 'false gaps', S['fa'], 'found on clean', S['hit'], S['tot'], 'mean usd per sheet', round(np.mean([r['usd'] for r in S['erased']]), 4), 'mean s', round(np.mean([r['seconds'] for r in S['erased']]), 1))

# duplicate sheet: best NCC of ink maps within +-12 px, pair by pair
def d(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    I = np.clip(cv2.medianBlur(g, 31).astype(np.float32) - g, 0, 255)
    return cv2.GaussianBlur(cv2.resize(I, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA), (0, 0), 1.2)
rows = {}
for b in SEEDS:
    fr, names = load_stack(b); M = [d(f) for f in fr]; pad = 12
    best = max(float(cv2.matchTemplate(M[i], M[j][pad:-pad, pad:-pad], cv2.TM_CCOEFF_NORMED).max()) for i, j in itertools.combinations(range(len(M)), 2))
    dup = [float(cv2.matchTemplate(M[i], d(rescan(fr[i], 3 + i))[pad:-pad, pad:-pad], cv2.TM_CCOEFF_NORMED).max()) for i in range(len(M))]
    rows[b] = dict(max_true_pair=round(best, 3), dup_min=round(min(dup), 3), dup_n=len(dup))
    print(b, rows[b])

