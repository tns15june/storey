"""Untouched stacks: track every seed and print presence per floor."""
import sys
sys.path.insert(0, 'lab/defects')
from track import *
from seeds import SEEDS
from concurrent.futures import ThreadPoolExecutor
def one(b):
    fr, names = load_stack(b)
    s = SEEDS[b]; obj = {k: v[1] for k, v in s['objects'].items()}
    r = run(fr, obj, f'clean_{b}', names.index(s['floor']), names)
    grid(fr, r, names, obj, OUT / 'thumbs' / f'grid_clean_{b}.jpg')
    return b, names, r
with ThreadPoolExecutor(5) as ex:
    for b, names, r in ex.map(one, list(SEEDS)):
        print(b, names)
        for o in r:
            print('  ', o, [(None if r[o][n] is None else (r[o][n]['area'], r[o][n]['conf'])) for n in names])
