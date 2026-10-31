"""Step 4. Blind Spark crop question at a place on one floor. The prompt never mentions other floors or what the comparison said. Answers are cached by sheet file and place, rounded to 30 px, so a planted stack reuses them wherever the sheet is unchanged."""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/presence")
import pres
from pres import OUT
import metalab as m

SIDE, UP = 260, 3
ANS = {"type": "object", "additionalProperties": False, "required": ["answer", "seen"],
       "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unclear"]}, "seen": {"type": "string"}}}
Q = {"stair": ("This is a crop from one floor plan of a building. Is a staircase drawn at or near the centre of this crop? A staircase is a run of closely spaced parallel lines (treads), straight or curved, "
               "often with an arrow or a break line. Walls, doors, hatching, dimension lines and text alone do not count. Answer yes, no or unclear, and say in a few words what is at the centre."),
     "shaft": ("This is a crop from one floor plan of a building. Is an elevator, lift or vertical shaft drawn at or near the centre of this crop? It is usually a small box with a diagonal cross, or a box labelled lift, elevator or shaft, or a round flue. "
               "An ordinary room does not count. Answer yes, no or unclear, and say in a few words what is at the centre.")}
C = OUT / "crops"
C.mkdir(exist_ok=True)


def key_of(path, loc, fam):
    x, y = int(round(loc[0] / 30)) * 30, int(round(loc[1] / 30)) * 30
    return hashlib.sha1(f"{path}|{x}|{y}|{fam}".encode()).hexdigest()[:16], x, y


def cut(path, x, y):
    im = cv2.imread(str(path))
    r = SIDE // 2
    P = cv2.copyMakeBorder(im, r, r, r, r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    c = P[y:y + SIDE, x:x + SIDE]
    return cv2.resize(c, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)


def ask(path, loc, fam, near=None):
    k, x, y = key_of(path, loc, fam)
    p = C / f"{k}.json"
    if p.exists():
        return dict(json.load(open(p)), cached=True)
    for o in (near or {}).get((str(path), fam), ()):  # a crop within 30 px of this place on the same sheet shows the same thing
        if abs(o["x"] - loc[0]) <= 30 and abs(o["y"] - loc[1]) <= 30:
            return dict(o, cached=True)
    r, info = m.spark(Q[fam], [cut(path, x, y)], schema=ANS, effort="low", max_tokens=4000)
    o = dict(answer=r["answer"], seen=r["seen"], usd=info["usd"], seconds=info["seconds"], x=x, y=y, fam=fam, path=str(path))
    json.dump(o, open(p, "w"), indent=1)
    return o


def ask_many(jobs, workers=6):
    """jobs: (path, loc, fam). Returns answers in order, in parallel, each cached."""
    near = {}
    for q in C.glob("*.json"):
        o = json.load(open(q))
        near.setdefault((o["path"], o["fam"]), []).append(o)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: ask(*j, near), jobs))
