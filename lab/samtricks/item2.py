"""Item 2: concept words on whole sheets. 8 words on 12 sheets, cached. Usage: item2.py run | score"""
import json
import sys

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/samtricks")
import api as a  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import reg  # noqa: E402

T = json.load(open(reg.TRUTH / "item2_words.json"))["truth"]
C = json.load(open(reg.TRUTH / "item4_clouds.json"))["clouds"]
OUT = reg.OUT / "item2.json"


def truth(k, w):
    t = T[k].get(w)
    if t == "clouds":
        return [c["grid"] for c in C[k]], []
    return (t["req"], t.get("opt", [])) if t else ([], [])


def present_state(k, w):
    """present, none (truly absent) or optional (only opt boxes, so neither right nor absent)."""
    req, opt = truth(k, w)
    return "present" if req else "optional" if opt else "none"


def inside(b, boxes):
    x, y = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return any(q[0] <= x <= q[2] and q[1] <= y <= q[3] for q in boxes)


def run():
    jobs = [(k, w) for k in reg.WORD_SHEETS for w in reg.WORDS]

    def one(j):
        k, w = j
        im = reg.sheet(k)
        h, wd = im.shape[:2]
        return k, w, [dict(box=a.to_grid(r["box"], wd, h), conf=r["conf"], area=r["area"]) for r in a.sam(w, im)]
    res = {}
    for k, w, rows in a.pmap(one, jobs, 8):
        res[f"{k}/{w}"] = rows
    OUT.write_text(json.dumps(res))
    print(len(res), "calls, spent", round(a.m.spent("samtricks"), 3))


def score():
    res = json.load(open(OUT))
    out = {}
    print("word: present right/any of n | absent sheets false of n | masks | verdict")
    for w in reg.WORDS:
        pres = [k for k in reg.WORD_SHEETS if present_state(k, w) == "present"]
        none = [k for k in reg.WORD_SHEETS if present_state(k, w) == "none"]
        right, anyh, false_abs, falses, masks = 0, 0, 0, 0, 0
        bad = []
        for k in reg.WORD_SHEETS:
            rows = res[f"{k}/{w}"]
            req, opt = truth(k, w)
            masks += len(rows)
            ranked = sorted(rows, key=lambda r: -(r["conf"] or 0))
            fal = [r for r in rows if not inside(r["box"], req + opt)]
            falses += len(fal)
            if k in pres:
                ok = bool(ranked) and inside(ranked[0]["box"], req)
                right += ok
                anyh += any(inside(r["box"], req) for r in rows)
                if not ok:
                    bad.append(f"{k}:{'none' if not rows else 'top wrong'}")
            if k in none and fal:
                false_abs += 1
                bad.append(f"{k}:false {len(fal)}")
        n, m = len(pres), len(none)
        rate = right / n if n else None
        v = "n/a" if not n else "keep" if rate >= .75 and false_abs <= 1 else "needs more" if rate >= .5 else "drop"
        out[w] = dict(present=n, right=right, any=anyh, absent=m, false_on_absent=false_abs, false_masks=falses, masks=masks, verdict=v, bad=bad)
        print(f"{w:15s} {right}/{anyh} of {n} | {false_abs} of {m} | masks {masks} false masks {falses} | {v} | {bad}")
    # baselines
    strip = [[900, 0, 1000, 1000]]
    pres = [k for k in reg.WORD_SHEETS if present_state(k, "title block") == "present"]
    print("baseline always-absent: right 0 on present, false 0 on absent, for every word")
    print("baseline title block fixed strip x>=900, centre in truth:", sum(inside(strip[0], truth(k, "title block")[0]) for k in pres), "of", len(pres))
    (reg.OUT / "item2_score.json").write_text(json.dumps(out, indent=1))
    for w in reg.WORDS:
        cells = []
        for k in reg.WORD_SHEETS:
            im = reg.sheet(k).copy()
            h, wd = im.shape[:2]
            req, opt = truth(k, w)
            for b in req:
                x = a.to_px(b, wd, h)
                cv2.rectangle(im, x[:2], x[2:], (255, 0, 255), 3)
            for b in opt:
                x = a.to_px(b, wd, h)
                cv2.rectangle(im, x[:2], x[2:], (255, 0, 255), 1)
            for r in res[f"{k}/{w}"]:
                x = a.to_px(r["box"], wd, h)
                cv2.rectangle(im, x[:2], x[2:], (0, 170, 0), 5)
            c = cv2.resize(im, (480, 340), interpolation=cv2.INTER_AREA)
            cv2.putText(c, f"{k} {w} {len(res[f'{k}/{w}'])}", (6, 20), 0, .6, (0, 0, 255), 2)
            cells.append(c)
        cv2.imwrite(str(reg.OUT / f"item2_{w.replace(' ', '_')}.jpg"), np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, 12, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 78])


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run()
    score()
