"""Spark box to SAM box prompt, on the raw drawing and on a cleaned drawing. Cached. usage: sam_rooms.py n"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'lab'); sys.path.insert(0, 'lab/walls')
import cv2, numpy as np
import metalab as m
import cc, plan as P, score as S, ask_spark as A
OUT = 'data/lab/walls/sam/'
os.makedirs(OUT, exist_ok=True)


def clean(img, u):
    """Cleaned drawing: only wall blocks, black on white, everything else gone."""
    w = P.walls(img, u, dbl_k=0.08)
    out = np.full(img.shape, 255, np.uint8)
    out[w] = 0
    return out


def segment(im, boxes):
    """One SAM call per 12 boxes. Returns list of bool masks the size of `im`, or None per box."""
    h, w = im.shape[:2]
    out = {}
    for i in range(0, len(boxes), 12):
        toks = []
        for j, b in enumerate(boxes[i:i + 12], i + 1):
            x1, y1, x2, y2 = [int(v) for v in (b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000)]
            toks.append(f'{j}<|box;x1={x1};y1={y1};x2={min(x2, w - 1)};y2={min(y2, h - 1)};w={w};h={h}|>')
        for r in m.sam_image('<0f>' + ','.join(toks), im):
            mask = np.zeros((h, w), bool)
            b = r.bounds
            mask[b.top:b.bottom, b.left:b.right] = m.raster(r)
            out[r.object_id] = (mask, float(r.confidence or 0))
    return out


def one(h):
    f = OUT + h.replace('/', '_') + '.npz'
    if os.path.exists(f):
        return f
    sp = A.one(h)
    d = cc.load(h)
    g = S.gt_setup(d)
    H, W = d['h'], d['w']
    im = A.fit(d['img'])
    boxes = [r['box'] for r in sp['rooms'] if len(r['box']) == 4 and r['box'][2] > r['box'][0] and r['box'][3] > r['box'][1]]
    res = {}
    for name, pic in (('raw', im), ('clean', A.fit(clean(d['img'], g['u'])))):
        ms = segment(pic, boxes)
        lab = np.zeros((H, W), np.int32)
        conf = {}
        # paint big rooms first so a small room stays on top
        for k, (mk, c) in sorted(ms.items(), key=lambda kv: -kv[1][0].sum()):
            full = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            lab[full] = k
            conf[k] = c
        res[name] = lab
        res[name + '_conf'] = np.array(sorted(conf.values()))
    np.savez_compressed(f, **res)
    return f


if __name__ == '__main__':
    n = int(sys.argv[1])
    hs = cc.split('test')
    rng = np.random.default_rng(1)
    hs = [hs[i] for i in rng.permutation(len(hs))[:n]]
    with ThreadPoolExecutor(6) as p:
        fs = list(p.map(one, hs))
    print(len(fs), 'spent', m.spent('walls'))
