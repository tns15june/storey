"""Contact sheets of every SAM box at conf>=.5 that does not sit on a truth stair, for labelling by eye. No API."""
import json, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed")
from common import *
A = ROOT / "data/lab/audit"; O = ROOT / "data/lab/agent/mine"; O.mkdir(parents=True, exist_ok=True)
T = json.load(open(ROOT / "lab/audit/truth.json")); NF = {"habs_va_bldg1": 5, "bme_k": 4, "wfu_huffman": 4}
cands = []
for b in NF:
    tr = json.load(open(A / "sam" / f"{b}__union.json"))["tracks"]
    for oid, o in sorted(tr.items(), key=lambda x: int(x[0])):
        for f in range(NF[b]):
            mk = o.get(str(f))
            if not mk or (mk["conf"] or 0) < .5 or mk["area"] <= 0:
                continue
            on = [k for k, s in T[b].items() if str(f) in s["boxes"] and hit(mk["box"], s["boxes"][str(f)], .25)]
            cands.append(dict(b=b, oid=int(oid), f=f, box=mk["box"], conf=mk["conf"], on=on))
json.dump(cands, open(O / "stair_cands.json", "w"), indent=0)
ghost = [c for c in cands if not c["on"]]
print(len(cands), "boxes,", len(ghost), "not on truth stair")
def tile(c, R=170, s=2.0):
    im = cv2.imread(str(A / c["b"] / f"f{c['f']}.png")); t = c["box"]
    cx, cy = (t[0] + t[2]) // 2, (t[1] + t[3]) // 2
    x0, y0 = max(cx - R, 0), max(cy - R, 0)
    v = cv2.resize(im[y0:cy + R, x0:cx + R], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    P = lambda x, y: (int((x - x0) * s), int((y - y0) * s))
    cv2.rectangle(v, P(t[0], t[1]), P(t[2], t[3]), (255, 0, 255), 2)
    cv2.putText(v, f"{c['b'][:6]} o{c['oid']} f{c['f']} {c['conf']}", (4, 16), 0, .5, (0, 0, 255), 1)
    return cv2.copyMakeBorder(v, 0, 680 - v.shape[0], 0, 680 - v.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))[:680, :680]
for i in range(0, len(ghost), 9):
    g = [tile(c) for c in ghost[i:i + 9]]
    while len(g) < 9: g.append(np.full((680, 680, 3), 255, np.uint8))
    cv2.imwrite(str(O / f"ghost_{i // 9}.jpg"), np.vstack([np.hstack(g[k:k + 3]) for k in (0, 3, 6)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print([ (c['b'][:4],c['oid'],c['f'],c['conf']) for c in ghost])
