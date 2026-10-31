"""Spark door boxes on zoomed tiles, then carve them into the blocker mask. CubiCasa test plans. usage: tile_doors.py n"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab'); sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/walls')
import cv2, numpy as np
import metalab as m
import cc, plan as P, score as S
OUT = '/Users/tsharma/Github/Meta/data/lab/walls/tiles/'
os.makedirs(OUT, exist_ok=True)
INT = {'type': 'integer'}
SC = {'type': 'object', 'additionalProperties': False, 'required': ['doors'], 'properties': {'doors': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['box'], 'properties': {'box': {'type': 'array', 'items': INT}}}}}}
PR = 'This is a zoomed part of a building floor plan. Find every door. A door is drawn as a quarter-circle swing arc with a straight leaf at a gap in a wall, or a plain gap between two rooms. Return a box [x1,y1,x2,y2] around each door symbol including its arc and its wall gap, on a 0-1000 grid, x then y. Ignore windows and furniture.'


def tiles(img, n=2, over=0.15):
    ink = P.ink_mask(img, 200)
    ys, xs = np.nonzero(ink)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    W, H = (x1 - x0) / (n - (n - 1) * over), (y1 - y0) / (n - (n - 1) * over)
    for j in range(n):
        for i in range(n):
            a, b = int(x0 + i * W * (1 - over)), int(y0 + j * H * (1 - over))
            yield a, b, int(a + W), int(b + H)


def one(h):
    f = OUT + h.replace('/', '_') + '.json'
    if os.path.exists(f):
        return json.load(open(f))
    d = cc.load(h)
    boxes, usd = [], 0.0
    def call(t):
        x0, y0, x1, y1 = t
        crop = d['img'][y0:y1, x0:x1]
        s = min(1.5, 1600 / max(crop.shape[:2]))
        r, i = m.spark(PR, [cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)], schema=SC, effort='low')
        w, h_ = x1 - x0, y1 - y0
        return [[x0 + b['box'][0] * w / 1000, y0 + b['box'][1] * h_ / 1000, x0 + b['box'][2] * w / 1000, y0 + b['box'][3] * h_ / 1000] for b in r['doors'] if len(b['box']) == 4], i['usd']
    for t in tiles(d['img']):
        b, u = call(t); boxes += b; usd += u
    out = dict(house=h, boxes=boxes, usd=usd)
    json.dump(out, open(f, 'w'))
    return out


if __name__ == '__main__':
    n = int(sys.argv[1])
    hs = cc.split('test')
    rng = np.random.default_rng(1)
    hs = [hs[i] for i in rng.permutation(len(hs))[:n]]
    with ThreadPoolExecutor(5) as p:
        rs = list(p.map(one, hs))
    print('usd', sum(r['usd'] for r in rs), 'boxes', sum(len(r['boxes']) for r in rs))
