"""Spark room boxes and door points on CubiCasa test plans. Cached per plan; usage: ask_spark.py n"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'lab'); sys.path.insert(0, 'lab/walls')
import cv2, numpy as np
import metalab as m
import cc
OUT = 'data/lab/walls/spark/'
os.makedirs(OUT, exist_ok=True)
INT = {'type': 'integer'}
ROOM = {'type': 'object', 'additionalProperties': False, 'required': ['rooms'], 'properties': {'rooms': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['label', 'box'], 'properties': {'label': {'type': 'string'}, 'box': {'type': 'array', 'items': INT}}}}}}
DOOR = {'type': 'object', 'additionalProperties': False, 'required': ['doors'], 'properties': {'doors': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['point', 'kind'], 'properties': {'point': {'type': 'array', 'items': INT}, 'kind': {'type': 'string', 'enum': ['interior', 'exterior']}}}}}}
RP = 'This is a building floor plan. Find every enclosed room and corridor inside the walls. Return a tight box [x1,y1,x2,y2] for each one on a 0-1000 grid, x then y, with the box edges on the inner faces of the room walls. Do not include balconies, terraces, stairs drawn outside, dimension text or furniture.'
DP = 'This is a building floor plan. Point at every door and every doorless opening in a wall between two spaces. A door is usually a gap in a wall with a quarter-circle swing arc. Return the point [x,y] in the middle of the wall gap, on a 0-1000 grid, x then y. Mark kind exterior if it leads out of the building or apartment. Windows are not doors.'


def fit(img, side=1800):
    s = min(1.0, side / max(img.shape[:2]))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def one(h):
    f = OUT + h.replace('/', '_') + '.json'
    if os.path.exists(f):
        return json.load(open(f))
    d = cc.load(h)
    im = fit(d['img'])
    r1, i1 = m.spark(RP, [im], schema=ROOM, effort='medium')
    r2, i2 = m.spark(DP, [im], schema=DOOR, effort='medium')
    out = dict(house=h, rooms=r1['rooms'], doors=r2['doors'], usd=i1['usd'] + i2['usd'], sec=[i1['seconds'], i2['seconds']], tok=[i1['output'], i2['output']])
    json.dump(out, open(f, 'w'))
    return out


if __name__ == '__main__':
    n = int(sys.argv[1])
    hs = cc.split('test')
    rng = np.random.default_rng(1)
    hs = [hs[i] for i in rng.permutation(len(hs))[:n]]
    with ThreadPoolExecutor(6) as p:
        rs = list(p.map(one, hs))
    print('usd', sum(r['usd'] for r in rs), 'sec', np.mean([r['sec'] for r in rs], 0), 'rooms', sum(len(r['rooms']) for r in rs), 'doors', sum(len(r['doors']) for r in rs))
