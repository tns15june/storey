"""Does the wall-pen and arc recipe hold on the other vector sets? Pen table, arc count, overlay of the heaviest pen."""
import sys, json
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/walls')
import collections
import cv2, fitz, numpy as np, vec
D = '/Users/tsharma/Github/Meta/data/plans/'
O = '/Users/tsharma/Github/Meta/data/lab/walls/'
Q = 72 * 0.25 / 0.3048   # 1/4 in = 1 ft in points per metre
SETS = {'mml_capitol': ('THE-CAPITOL.pdf', 3, Q), 'mml_baxter': (None, 3, Q), 'mml_abbot': (None, 3, Q), 'usc_leconte': ('USC_LeConte_Bid_Drawings.pdf', 2, Q), 'carpinteria_adu1': ('plan1.pdf', 7, Q), 'wfu_efird': ('Efird.pdf', 1, None), 'brandeis_hall': (None, 0, Q)}
cat = {e['id']: e for e in json.load(open(D + 'catalog.json'))}
rows = {}
for k, (f, pg, q) in SETS.items():
    e = cat[k]
    fl = e['floors'][min(1, len(e['floors']) - 1)]
    f, pg = fl['file'], fl['page']
    g = fitz.open(D + k + '/' + f)[pg]
    s = vec.strokes(g)
    tot = {w: float(np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]).sum()) for w, a in s.items()}
    lng = {w: float(np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])[np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]) > 8].sum()) for w, a in s.items()}
    fills = sum(1 for x in g.get_drawings() if x['type'] != 's')
    arcs = vec.door_arcs(g, q) if q else []
    print(k, f[:30], 'page', pg, g.rect.width, g.rect.height, 'images', len(g.get_images()), 'fills', fills, 'arcs', len(arcs))
    print('   pens', sorted([(w, round(v), round(lng[w])) for w, v in tot.items()], key=lambda r: -r[0])[:9])
    rows[k] = (g, s)
    sc = 1600 / max(g.rect.width, g.rect.height)
    pens = sorted(s, reverse=True)
    pens = [p for p in pens if lng[p] > 0.08 * sum(lng.values())][:3]
    tiles = []
    for p in pens:
        im = np.full((int(g.rect.height * sc), int(g.rect.width * sc)), 255, np.uint8)
        im[vec.draw(s[p], (0, 0, g.rect.width, g.rect.height), sc, 1)] = 0
        cv2.putText(im, f'{k} pen {p}', (10, 30), 0, 1, 0, 2)
        tiles.append(im)
    cv2.imwrite(O + f'pens_{k}.jpg', np.hstack(tiles))
