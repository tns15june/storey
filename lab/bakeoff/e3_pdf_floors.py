"""E3: PDF input against rendered images, multi-sheet calls, prefix caching. E4: can one call compare floors? Usage: e3_pdf_floors.py pdf|floors|cache|table"""
import base64
import sys
import time

sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/bakeoff')
from common import *  # noqa
import cv2
import e1_meta as e1
import fitz
import numpy as np
import truth_meta as tm

S13, S12, W = 'muse-spark-1.3', 'muse-spark-1.2', 'muse-wonder-1.0'
DEV_PDF = ['n161', 'n79', 'u1', 'c3', 'n7', 'n93', 'a7', 'c2', 'u2']
CAP_TRUTH = "A1.0 A2.0 A2.1 A2.2 A2.3 A2.4 A3.0 A4.0 A5.0 A5.1 E0.1 E1.1 E2.1 E5.1 E7.1 P0.1 P1.1 P6.1 M0.1 M1.1 M6.1 L1.0".split()
USC_TRUTH = {0: 'G001', 1: 'G002', 2: 'G003', 3: 'AD101', 12: 'MD100', 15: 'E001'}
SET_S = obj(pages=arr(obj(page=INT, sheet_number=STR, sheet_title=STR, sheet_class=e1.SCHEMA['properties']['sheet_class'], scale=STR)))
SET_P = "This is a building drawing set. For every page, in order, give page (1 for the first page), the sheet_number printed in the title block, sheet_title as printed, sheet_class, and scale, the scale of the main drawing copied exactly (empty if AS SHOWN, NTS or none). Do not skip any page."


def pdf_call(exp, key, cfg, prompt, data, schema, model=S13, effort='low', name='sheet.pdf'):
    k = (exp, key, cfg, 0)
    if k in _done():
        return _done()[k]
    m._guard()
    url = 'data:application/pdf;base64,' + base64.b64encode(data).decode()
    t0 = time.time()
    r = m._retry(lambda: m.client.chat.completions.create(model=model, reasoning_effort=effort, max_completion_tokens=24000,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "file", "file": {"filename": name, "file_data": url}}]}],
        response_format={"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}}))
    u, pi, pc, po = r.usage, *m.PRICES[model]
    c = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
    usd = ((u.prompt_tokens - c) * pi + c * pc + u.completion_tokens * po) / 1e6
    m._log(model, usd, input=u.prompt_tokens, output=u.completion_tokens, kind='pdf')
    row = dict(exp=exp, key=key, cfg=cfg, run=0, model=model, effort=effort, out=json.loads(r.choices[0].message.content), usd=usd,
               seconds=time.time() - t0, input=u.prompt_tokens, output=u.completion_tokens, cached=c)
    save(row)
    return row


def _done():
    import common
    return common._done


def one_page(key):
    f, p = s.ALL[key]
    src = fitz.open(s.P / f)
    d = fitz.open()
    d.insert_pdf(src, from_page=p, to_page=p)
    return d.tobytes()


def run_pdf():
    jobs = [lambda k=k: pdf_call('pdfmeta', k, 'pdf-low', e1.PROMPT, one_page(k), e1.SCHEMA) for k in DEV_PDF]
    jobs.append(lambda: pdf_call('pdfset', 'capitol', 'pdf-low', SET_P, open(s.P / s.CAP, 'rb').read(), SET_S))
    jobs.append(lambda: pdf_call('pdfset', 'usc', 'pdf-low', SET_P, open(s.P / s.USC, 'rb').read(), SET_S))
    pool(jobs, 4)
    d = fitz.open(s.P / s.CAP)
    ims = lambda: [cv2.imdecode(np.frombuffer(p.get_pixmap(matrix=fitz.Matrix(2000 / p.rect.width, 2000 / p.rect.width), alpha=False).tobytes('png'), np.uint8), 1) for p in d]
    call('pdfset', 'capitol', 'img-2k-1call', 0, SET_P, ims, SET_S, S13, 'low')


def norm(x):
    return tm.norm_sheet(x)


def score_set(rec, truth):
    pages = rec['out']['pages']
    byp = {p['page']: p for p in pages}
    items = truth.items() if isinstance(truth, dict) else enumerate(truth)
    hit = [norm(byp.get(i + 1, {}).get('sheet_number')) == norm(t) for i, t in items]
    return sum(hit), len(hit), len(pages)


def table_pdf():
    print('single page: PDF versus image, same prompt, sheets', DEV_PDF)
    for exp, cfg in (('pdfmeta', 'pdf-low'), ('meta', 's13-low-2k')):
        r = [x for x in rows(exp) if x['cfg'] == cfg and x['key'] in DEV_PDF and x['run'] == 0]
        sc = [tm.score(x['key'], x['out']) for x in r]
        print(exp, cfg, 'n', len(r), {f: sum(q[f] for q in sc) for f in ('cls', 'num', 'level', 'scale')}, f"usd/sheet {sum(x['usd'] for x in r)/len(r):.4f} s {sum(x['seconds'] for x in r)/len(r):.1f} in {sum(x['input'] for x in r)/len(r):.0f} out {sum(x['output'] for x in r)/len(r):.0f}")
    for r in rows('pdfmeta'):
        print('  ', r['key'], r['input'], r['output'], r['out']['sheet_number'], '|', r['out']['level'], '|', r['out']['scale'])
    for r in rows('pdfset'):
        t = CAP_TRUTH if r['key'] == 'capitol' else USC_TRUTH
        print(r['key'], r['cfg'], 'sheet numbers right, of, pages returned', score_set(r, t), f"usd {r['usd']:.4f} s {r['seconds']:.0f} in {r['input']} out {r['output']}")


# ---- floors ----
FLOOR_S = obj(stairs=arr(obj(name=STR, floors=arr(obj(floor=STR, present={"type": "boolean"}, x=INT, y=INT)))))
FLOOR_P = """These images are floor plans of one building, one image per floor, in this order: {order}. Staircases in the same place on consecutive floors are the same staircase.
List every distinct staircase in the building. For each give a short name, and for every floor in the list say whether it is drawn on that floor (present) and its centre x, y in the 0-1000 range of that floor's image (0, 0 if absent). A stair drawn struck through or filled in counts as absent. Compare the floors carefully, stair by stair."""
BF = {'-1': 'bme_f/bme_-1.jpg', '0': 'bme_f/bme_0.jpg', '1': 'bme_f/bme_1.jpg', '2': 'bme_f/bme_2.jpg'}
FLOORSETS = {
    'bme_f': dict(order='basement, ground floor, first floor, second floor', ims=['-1', '0', '1', '2'], ref=1,
                  truth={'I': ((446, 640), [1, 1, 1, 1]), 'II': ((602, 476), [1, 1, 1, 1]), 'III': ((294, 471), [1, 1, 1, 1]), 'NW': ((236, 172), [1, 1, 1, 0]), 'NE': ((661, 179), [None, 1, 1, 0])}),
    'nibs': dict(order='first floor, second floor', ims=['n161', 'n162'], ref=0, truth={'1CS3': ((410, 168), [1, 1]), '1AS1': ((538, 293), [1, 1]), '1BS2': ((550, 562), [1, 1])}),
    'capitol': dict(order='ground floor, main floor, upper floor', ims=['c2', 'c3', 'c4'], ref=1,
                    truth={'interior': ((690, 505), [0, 1, 1]), 'north ext': ((768, 153), [1, 1, None]), 'south ext': ((768, 686), [1, 1, None])}),
}


def floor_images(b):
    fs = FLOORSETS[b]
    if b == 'bme_f':
        return lambda: [s.limit(cv2.imread(str(s.P / BF[i])), 2000) for i in fs['ims']]
    return lambda: [s.full(i, 2000) for i in fs['ims']]


def run_floors():
    jobs = []
    for cfg, model, eff, runs in (('s13-med', S13, 'medium', (0, 1, 2)), ('s12-med', S12, 'medium', (0,)), ('w-med', W, 'medium', (0,))):
        for b in FLOORSETS:
            for r in runs:
                jobs.append(lambda b=b, cfg=cfg, model=model, eff=eff, r=r: call('floors', b, cfg, r, FLOOR_P.format(order=FLOORSETS[b]['order']), floor_images(b), FLOOR_S, model, eff))
    pool(jobs, 5)


def score_floors(b, out):
    fs = FLOORSETS[b]
    n = len(fs['ims'])
    ok = tot = 0
    used, detail = set(), []
    for name, (pos, want) in fs['truth'].items():
        best, bd = None, 1e9
        for i, st in enumerate(out['stairs']):
            if i in used or len(st['floors']) != n:
                continue
            f = st['floors'][fs['ref']]
            if f['present']:
                d = np.hypot(f['x'] - pos[0], f['y'] - pos[1])
                if d < bd:
                    best, bd = i, d
        got = None
        if best is not None and bd < 90:
            used.add(best)
            got = [int(f['present']) for f in out['stairs'][best]['floors']]
        for j, w in enumerate(want):
            if w is None:
                continue
            tot += 1
            ok += (got is not None and got[j] == w)
        detail.append(f"{name}:{got}")
    return ok, tot, len(out['stairs']), detail


def table_floors():
    for r in sorted(rows('floors'), key=lambda x: (x['key'], x['cfg'], x['run'])):
        ok, tot, n, d = score_floors(r['key'], r['out'])
        print(f"{r['key']:8}{r['cfg']:9}run{r['run']} pairs {ok}/{tot} stairs listed {n} ${r['usd']:.4f} {r['seconds']:.0f}s in {r['input']} out {r['output']} {d}")


# ---- prefix cache: image first, then two different questions ----
def run_cache():
    im = m.data_url(s.full('c3', 2000))
    res = []
    for q in ('What is the sheet number? Answer with the number only.', 'What is the scale of the main floor plan? Answer with the scale only.', 'What is the sheet title? Answer with the title only.'):
        m._guard()
        t0 = time.time()
        r = m._retry(lambda: m.client.chat.completions.create(model=S13, reasoning_effort='low', prompt_cache_key='bakeoff-sheet', max_completion_tokens=2000,
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": im}}, {"type": "text", "text": q}]}]))
        u = r.usage
        c = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
        pi, pc, po = m.PRICES[S13]
        usd = ((u.prompt_tokens - c) * pi + c * pc + u.completion_tokens * po) / 1e6
        m._log(S13, usd, input=u.prompt_tokens, output=u.completion_tokens, kind='cachetest')
        res.append(dict(q=q[:30], input=u.prompt_tokens, cached=c, output=u.completion_tokens, usd=round(usd, 5), s=round(time.time() - t0, 1)))
        print(res[-1], flush=True)
    json.dump(res, open(s.OUT / 'cache.json', 'w'), indent=1)


if __name__ == '__main__':
    import numpy as np
    what = sys.argv[1] if len(sys.argv) > 1 else 'table'
    if what == 'pdf':
        run_pdf()
    if what == 'floors':
        run_floors()
    if what == 'cache':
        run_cache()
    table_pdf(); table_floors()
